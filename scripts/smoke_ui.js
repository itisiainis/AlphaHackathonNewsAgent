#!/usr/bin/env node
/* Дымовая проверка интерфейса на живом сервере (B03, шаг 6): повтор сценариев аудита, которые чинил fc4b14c.

   Запуск (сервер уже поднят — лучше на чистом APP_CACHE_DIR и с заглушкой модели вместо ключа):
     NODE_PATH=/opt/node-tools/node_modules node scripts/smoke_ui.js http://127.0.0.1:8000 /tmp/smoke-ui [--no-model]

   Проверки: ссылка «Переслать» открывает ту же компанию, роль и карточку — и у демо-компании, и у любой другой;
   ссылка на пример команды открывается; быстрое переключение компаний не рассинхронизирует список и ленту;
   вставленный текст → «Полный разбор ИИ» доходит до «Разбор готов»; в консоли нет ошибок, кроме ожидаемых 401/404/429.
   Снимки экрана — в каталоге вывода; внутри репозитория он лежать не может. Итог — JSON на stdout; код выхода 1,
   если не прошла хотя бы одна проверка (её имя и причина — в stderr), 2 — неверные аргументы.
   Вставленный текст остаётся в ленте демо-компании как присланный клиентом — запускайте на проверочном сервере,
   не на том, что покажете жюри. --no-model пропускает «Полный разбор ИИ»: он запускает разбор на модели сервера
   (с заглушкой — 0 ₽). Браузер: CHROMIUM_PATH, по умолчанию Chromium из /opt/pw-browsers. */
const crypto = require('crypto');
const fs = require('fs');
const path = require('path');
const { chromium } = require('playwright');

const ROOT = path.resolve(__dirname, '..');
const CHROME = process.env.CHROMIUM_PATH || '/opt/pw-browsers/chromium-1194/chrome-linux/chrome';
const DEEP = { company: 'U_200890', role: 'treasurer', card: 'ev-11bd67b498-U_200890' };  // ссылка из «Переслать»
const SEED = 'i-steel';  // пример команды из frontend/mock_data.json
const PASTE = 'Банк России снизил ключевую ставку с 16% до 14% годовых';
const PASTE_COMPANIES = ['U_25835', 'U_177581', 'U_200890'];
const EXPECTED_HTTP = new Set([401, 404, 429]);  // нет карточки, неверный токен, лимит запросов к модели
const SWITCHES = 5, SWITCH_GAP_MS = 30;
const AI_TIMEOUT_MS = 150000;

const args = process.argv.slice(2);
const NO_MODEL = args.includes('--no-model');
const [BASE_ARG, OUT_ARG] = args.filter(a => !a.startsWith('--'));
if (!BASE_ARG || !OUT_ARG) {
  console.error('Использование: node scripts/smoke_ui.js <адрес сервера> <каталог для снимков> [--no-model]');
  process.exit(2);
}
const BASE = BASE_ARG.replace(/\/+$/, '');
const OUT = path.resolve(OUT_ARG);
if (OUT === ROOT || OUT.startsWith(ROOT + path.sep)) {
  console.error(`Каталог для снимков не может быть внутри репозитория (${ROOT}) — укажите, например, /tmp/smoke-ui`);
  process.exit(2);
}
fs.mkdirSync(OUT, { recursive: true });

const checks = [], consoleErrors = [];
const sha1 = s => crypto.createHash('sha1').update(s, 'utf8').digest('hex');
class Skip extends Error {}
function must(cond, why) { if (!cond) throw new Error(why); }

async function getJson(url) {
  const r = await fetch(BASE + url);
  if (!r.ok) throw new Error(`GET ${url} → HTTP ${r.status}`);
  return r.json();
}

// Каждая проверка — в новом контексте браузера: холодная загрузка, пустое хранилище, как у коллеги по ссылке
async function run(browser, name, fn) {
  const ctx = await browser.newContext({ viewport: { width: 1366, height: 900 }, locale: 'ru-RU' });
  // все уведомления за проверку: они исчезают через несколько секунд, а опрос страницы мог бы их пропустить
  await ctx.addInitScript(() => {
    window.__smokeToasts = [];
    new MutationObserver(ms => ms.forEach(m => m.addedNodes.forEach(n => {
      if (n.nodeType === 1 && n.classList.contains('toast')) window.__smokeToasts.push(n.textContent);
    }))).observe(document, { childList: true, subtree: true });
  });
  const page = await ctx.newPage(), responses = [];
  page.on('console', m => {
    if (m.type() !== 'error') return;
    const code = +((m.text().match(/status of (\d{3})/) || [])[1] || 0);
    if (!EXPECTED_HTTP.has(code)) consoleErrors.push({ check: name, text: m.text(), url: m.location().url || '' });
  });
  page.on('pageerror', e => consoleErrors.push({ check: name, text: 'ошибка скрипта страницы: ' + e.message }));
  page.on('response', r => {
    responses.push({ url: r.url(), status: r.status(), method: r.request().method() });
    if (r.status() >= 500) consoleErrors.push({ check: name, text: `HTTP ${r.status()} ${r.request().method()} ${r.url()}` });
  });
  const shot = path.join(OUT, `${String(checks.length + 1).padStart(2, '0')}_${name}.png`);
  const t0 = Date.now(), res = { name, ok: true, status: 'pass', detail: '', ms: 0, shot };
  try {
    res.detail = (await fn(page, responses)) || '';
  } catch (e) {
    Object.assign(res, e instanceof Skip ? { status: 'skip' } : { ok: false, status: 'fail' }, { detail: e.message });
  }
  res.ms = Date.now() - t0;
  await page.screenshot({ path: shot }).catch(() => { res.shot = null; });
  await ctx.close();
  checks.push(res);
}

const view = page => page.evaluate(() => ({
  company: (document.querySelector('#sel-company') || {}).value,
  role: (document.querySelector('#sel-role') || {}).value,
  open: [...document.querySelectorAll('article.ic.open')].map(a => a.id.replace(/^card-/, '')),
  hash: location.hash,
}));

// 1. Ссылка «Переслать» несёт компанию и роль: у коллеги открывается та же карточка той же компании
async function deepLink(page) {
  await page.goto(`${BASE}/?c=${DEEP.company}&r=${DEEP.role}#feed/${DEEP.card}`);
  await page.waitForSelector(`[id="card-${DEEP.card}"].open`, { timeout: 15000 }).catch(() => {});
  const v = await view(page);
  must(v.company === DEEP.company, `в списке компаний ${v.company}, а не ${DEEP.company}`);
  must(v.role === DEEP.role, `роль ${v.role}, а не ${DEEP.role}`);
  must(v.open.includes(DEEP.card), `карточка ${DEEP.card} не раскрыта (раскрыты: ${v.open.join(', ') || 'нет'}, ${v.hash})`);
  return `${v.company} · ${v.role} · открыта ${DEEP.card}`;
}

// 2. Старая ссылка без компании на пример команды: компания определяется по карточке
async function seedLink(page) {
  const boot = await getJson('/api/bootstrap');
  const owner = Object.keys(boot.insights).find(c => boot.insights[c].some(i => i.id === SEED));
  must(owner, `карточки ${SEED} нет в /api/bootstrap`);
  await page.goto(`${BASE}/#feed/${SEED}`);
  await page.waitForSelector(`[id="card-${SEED}"].open`, { timeout: 15000 }).catch(() => {});
  const v = await view(page);
  must(v.open.includes(SEED), `карточка ${SEED} не раскрыта (раскрыты: ${v.open.join(', ') || 'нет'}, ${v.hash})`);
  must(v.company === owner, `карточка компании ${owner}, а в списке ${v.company}`);
  return `${owner} · открыта ${SEED}`;
}

// 3. Ссылка на разбор компании не из демо-набора: компания догружается по ссылке — и с ?c=&r=, и в старом виде,
//    где компания видна только по окончанию id карточки (#feed/<событие>-<компания>)
async function nonDemoLink(page) {
  const boot = await getJson('/api/bootstrap'), loaded = new Set(boot.companies.map(c => c.id));
  let cid = null, card = null;
  for (const c of boot.all_companies.map(x => x.id).filter(id => !loaded.has(id)).slice(0, 30)) {
    if (!(await getJson(`/api/companies/${c}/events?limit=1`)).length) continue;
    // так ленту открыл бы тот, кто пересылает ссылку: сервер готовит краткие разборы лучших событий
    const i = (await getJson(`/api/companies/${c}`)).insights.find(x => x.id.endsWith('-' + c));
    if (i) { cid = c; card = i.id; break; }
  }
  must(card, 'не нашлось компании вне демо-набора с разбором, на который можно сослаться');
  for (const [url, role] of [[`/?c=${cid}&r=ceo#feed/${card}`, 'ceo'], [`/#feed/${card}`, null]]) {
    await page.goto('about:blank');
    await page.goto(BASE + url);
    await page.waitForSelector(`[id="card-${card}"].open`, { timeout: 15000 }).catch(() => {});
    const v = await view(page);
    must(v.company === cid, `${url}: в списке компаний ${v.company}, а не ${cid}`);
    must(!role || v.role === role, `${url}: роль ${v.role}, а не ${role}`);
    must(v.open.includes(card), `${url}: карточка не раскрыта (раскрыты: ${v.open.join(', ') || 'нет'}, ${v.hash})`);
  }
  return `${cid} · открыта ${card} — по ссылке с ?c=&r= и по старой ссылке без них`;
}

// 4. Пять переключений компаний с промежутком 30 мс: в списке и в ленте в итоге одна и та же — последняя выбранная
async function fastSwitch(page, responses) {
  await page.goto(`${BASE}/#feed`);
  await page.waitForFunction(() => document.querySelectorAll('#sel-company option').length > 0, null, { timeout: 15000 });
  const opts = await page.$$eval('#sel-company option', o => o.map(x => x.value));
  // компании, уже загруженные в браузер (bootstrap самой страницы), выбираются без запроса к серверу — гонки нет
  const loaded = new Set(await page.evaluate(() => DATA.companies.map(c => c.id)));
  const cand = opts.filter(id => !loaded.has(id));
  must(cand.length >= SWITCHES, `для проверки нужно ${SWITCHES} компаний вне bootstrap, есть ${cand.length}`);
  // последняя — компания, у которой есть отобранные события: её лента точно состоит из карточек с её id
  let last = null;
  for (const id of cand.slice(-20).reverse()) {
    const ev = await getJson(`/api/companies/${id}/events?limit=1`);
    if (ev.length) { last = id; break; }
  }
  must(last, 'не нашлось компании с отобранными событиями для последнего переключения');
  // последнюю прогреваем на сервере (краткие разборы уже посчитаны): её ответ придёт раньше ответов по предыдущим —
  // ровно та гонка, из-за которой до fc4b14c в списке была одна компания, а в ленте другая
  await getJson(`/api/companies/${last}`);
  const rest = cand.filter(id => id !== last), step = Math.floor(rest.length / SWITCHES) || 1;
  const picks = [...Array(SWITCHES - 1).keys()].map(k => rest[k * step]).concat(last);
  await page.evaluate(async ([ids, gap]) => {
    const s = document.querySelector('#sel-company');
    for (const id of ids) {
      s.value = id; s.dispatchEvent(new Event('change', { bubbles: true }));
      await new Promise(r => setTimeout(r, gap));
    }
  }, [picks, SWITCH_GAP_MS]);
  // ждём ответы на загрузку всех пяти компаний — порядок их прихода и есть источник гонки
  const answered = () => picks.every(id => responses.some(r => r.url.endsWith(`/api/companies/${id}`)));
  for (const t0 = Date.now(); !answered() && Date.now() - t0 < 20000;) await page.waitForTimeout(100);
  must(answered(), 'сервер не ответил на загрузку всех компаний за 20 с');
  await page.waitForTimeout(800);
  const v = await page.evaluate(() => ({
    company: document.querySelector('#sel-company').value,
    owners: [...new Set([...document.querySelectorAll('article.ic[id^="card-"]')]
      .map(a => (a.id.match(/-(U_\d+)$/) || [])[1]).filter(Boolean))],
    state: typeof state !== 'undefined' ? state.company : null,
  }));
  const seq = picks.join(' → ');
  must(v.company === last, `в списке ${v.company}, а последней выбрали ${last} (${seq})`);
  must(v.owners.length, `в ленте ${last} нет карточек — не по чему сверить (${seq})`);
  must(v.owners.length === 1 && v.owners[0] === v.company, `в списке ${v.company}, а в ленте карточки ${v.owners.join(', ')} (${seq})`);
  must(v.state === null || v.state === v.company, `в списке ${v.company}, а интерфейс считает выбранной ${v.state} (${seq})`);
  return `${seq}: в списке и в ленте ${v.company}`;
}

// 5. Вставленный текст → краткий разбор → «Полный разбор ИИ»: раньше падал с «событие не найдено» (404)
async function pasteFullAi(page, responses) {
  if (NO_MODEL) throw new Skip('пропущено: --no-model');
  const boot = await getJson('/api/bootstrap');  // свежий: прежние проверки могли добавить карточки
  must(boot.status && boot.status.llm_online,
    'модель сервера недоступна — запустите сервер с заглушкой модели (LLM_BASE_URL, LLM_API_KEY) или передайте --no-model');
  // id события детерминирован (agent/pipeline.py: event_from_text). Где этот текст или тот же сюжет уже разобран,
  // вставка честно открывает готовый разбор без запуска ИИ — тогда пробуем следующую демо-компанию
  const title = PASTE.split(/[.!?\n]/)[0].slice(0, 200).trim();
  const eid = 'ev-' + sha1(`${title}\n${PASTE}`).slice(0, 10);
  const fresh = PASTE_COMPANIES.filter(c => !(boot.insights[c] || []).some(i => i.id === `${eid}-${c}` || i.title === title));
  const texts = () => page.evaluate(() => ({
    toasts: window.__smokeToasts || [],
    jobs: (document.querySelector('#jobs') || {}).innerText || '',
    page: document.body.innerText,
  }));
  const notFound = s => {  // кусок текста вокруг «событие не найдено» — в отчёт, а не вся страница
    const t = ([...s.toasts, s.jobs, s.page].find(x => /событие не найдено/i.test(x)) || '').replace(/\s+/g, ' ');
    const k = t.search(/событие не найдено/i);
    return k < 0 ? '' : t.slice(Math.max(0, k - 120), k + 120).trim();
  };
  let cid, s;
  const skipped = [];
  for (const c of fresh) {
    await page.goto(`${BASE}/?c=${c}#feed`);
    await page.waitForSelector('#quick-text', { state: 'attached', timeout: 15000 });
    await page.evaluate(() => { window.__smokeToasts = []; });
    await page.click('.quick-more summary');
    await page.fill('#quick-text', PASTE);
    await page.click('#quick-text-go');
    await page.waitForSelector('article.ic.open', { timeout: 15000 }).catch(() => {});
    await page.waitForTimeout(500);
    s = await texts();
    must(!notFound(s), `«событие не найдено» после «Оценить текст» (${c}): ${notFound(s)}`);
    const again = s.toasts.find(t => /уже разобран|уже разбирали/.test(t));
    if (!again) { cid = c; break; }
    skipped.push(`${c}: ${again}`);
  }
  must(cid, 'текст или тот же сюжет уже разобран на этом сервере для всех демо-компаний — запустите сервер'
    + ' на чистом APP_CACHE_DIR' + (skipped.length ? ` (${skipped.join('; ')})` : ''));
  must(await page.$('article.ic.open'), 'краткий разбор не открылся за 15 с' + (s.toasts.length ? ': ' + s.toasts.join(' | ') : ''));
  // после краткого разбора ИИ может запуститься сам — тогда кнопка «Полный разбор ИИ» показывает его ход
  const up = await page.$('article.ic.open [data-upgrade]');
  let how;
  if (up && await up.isEnabled()) { await up.click(); how = 'нажата «Полный разбор ИИ»'; }
  else if (await page.$('#jobs .job.running, #jobs .job.done')) how = 'полный разбор ИИ запустился сам после краткого';
  else throw new Error('полный разбор ИИ не запустился: нет ни кнопки «Полный разбор ИИ», ни задачи в углу экрана'
    + (s.toasts.length ? ` (уведомления: ${s.toasts.join(' | ')})` : ''));
  const t0 = Date.now();
  for (;;) {
    s = await texts();
    must(!notFound(s), `«событие не найдено»: ${notFound(s)}`);
    if (s.toasts.some(t => t.includes('Разбор готов'))) break;
    must(!/Разбор не получился/.test(s.jobs), 'разбор не получился: ' + s.jobs.replace(/\s+/g, ' ').trim());
    must(Date.now() - t0 < AI_TIMEOUT_MS, `нет «Разбор готов» за ${AI_TIMEOUT_MS / 1000} с (${s.jobs.replace(/\s+/g, ' ').trim()})`);
    await page.waitForTimeout(250);
  }
  const stream = responses.filter(r => r.method === 'POST' && /\/events\/[^/]+\/analyze\/stream$/.test(r.url));
  must(stream.length && stream.every(r => r.status === 200),
    'запрос полного разбора: ' + (stream.map(r => `HTTP ${r.status} ${r.url}`).join('; ') || 'не отправлялся'));
  const open = await page.$('[data-job-open]');
  if (open) { await open.click(); await page.waitForTimeout(800); }
  return `${cid}: ${how}; «${s.toasts.find(t => t.includes('Разбор готов'))}» через ${Math.round((Date.now() - t0) / 1000)} с`;
}

(async () => {
  let alive = false;
  try { alive = !!(await getJson('/api/bootstrap')); } catch (e) {
    checks.push({ name: 'server', ok: false, status: 'fail', detail: `сервер ${BASE} недоступен: ${e.message}` });
  }
  if (alive) {
    const browser = await chromium.launch({ executablePath: CHROME });
    try {
      await run(browser, 'deep_link_company_role', deepLink);
      await run(browser, 'deep_link_seed_card', seedLink);
      await run(browser, 'deep_link_non_demo', nonDemoLink);
      await run(browser, 'fast_company_switch', fastSwitch);
      await run(browser, 'paste_full_ai', pasteFullAi);
    } finally { await browser.close(); }
    checks.push({ name: 'console_clean', ok: !consoleErrors.length, status: consoleErrors.length ? 'fail' : 'pass',
      detail: consoleErrors.length ? `ошибок в консоли: ${consoleErrors.length}` : 'ошибок нет (кроме ожидаемых 401/404/429)' });
  }
  const failed = checks.filter(c => !c.ok).map(c => c.name);
  for (const c of checks.filter(x => !x.ok)) console.error(`FAIL ${c.name}: ${c.detail}`);
  for (const e of consoleErrors) console.error(`  консоль [${e.check}] ${e.text}${e.url ? ' @ ' + e.url : ''}`);
  console.log(JSON.stringify({ ok: !failed.length, base: BASE, out_dir: OUT, failed, checks, console_errors: consoleErrors }, null, 2));
  process.exit(failed.length ? 1 : 0);
})().catch(e => {
  console.error('FAIL smoke_ui: ' + (e && e.stack || e));
  console.log(JSON.stringify({ ok: false, base: BASE, out_dir: OUT, failed: ['smoke_ui'], checks, error: String(e && e.message || e) }));
  process.exit(1);
});
