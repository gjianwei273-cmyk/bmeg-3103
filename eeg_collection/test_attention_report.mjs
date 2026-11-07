// Run the actual bundled report script against a minimal DOM, without dependencies.
// This checks rendering logic and interactions, not browser layout/CSS.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import test from 'node:test';

const html = fs.readFileSync(process.argv[2], 'utf8');
const embedded = html.match(/id="reportData">([\s\S]*?)<\/script>/)[1];
const script = html.match(/<script>\s*([\s\S]*?)<\/script>/)[1];
const data = JSON.parse(embedded);

class Element {
  constructor(tag) { this.tag = tag; this.children = []; this.attributes = {}; this.style = {}; this.value = ''; this.events = {}; this.text = ''; }
  setAttribute(key, value) { this.attributes[key] = String(value); }
  appendChild(child) {
    this.children.push(child);
    if (this.tag === 'select' && this.children.length === 1 && this.value === '') this.value = String(child.value);
    return child;
  }
  append(...children) { children.forEach(child => this.appendChild(child)); }
  replaceChildren(...children) { this.children = []; this.text = ''; this.append(...children); }
  set textContent(value) { this.text = String(value); this.children = []; }
  get textContent() { return this.text + this.children.map(c => c.textContent).join(''); }
  addEventListener(event, fn) { this.events[event] = fn; }
}
const elements = new Map();
for (const match of html.matchAll(/<(\w+)[^>]*\bid="([^"]+)"/g)) elements.set(match[2], new Element(match[1]));
elements.get('reportData').textContent = embedded;
for (const [id, value] of Object.entries({ method: 'all', vertical: 'balanced_accuracy', traceConfig: 'heldout' })) elements.get(id).value = value;
const document = {
  getElementById(id) { assert.ok(elements.has(id), `Missing HTML element ${id}`); return elements.get(id); },
  createElement(tag) { return new Element(tag); },
  createElementNS(_ns, tag) { return new Element(tag); }
};
const context = vm.createContext({ document, console });
vm.runInContext(script, context);
const get = id => elements.get(id);
const descendants = element => [element, ...element.children.flatMap(descendants)];

test('renders validation values and every candidate without external dependencies', () => {
  assert.equal(get('overview').children.length, 4);
  assert.equal(get('validationRows').children.length, Object.keys(data.validation).length);
  assert.equal(get('tradeoff').children.filter(e => e.tag === 'circle').length, data.grid_size);
  assert.equal(get('heatmap').children.filter(e => e.tag === 'rect').length, data.grid_size);
  assert.equal(get('session').children.length, data.traces.length);
  assert.equal(get('foldRows').children.length, data.traces.length);
  assert.ok(!/<script[^>]+src=/.test(html));
});

test('method and score controls redraw finite charts', () => {
  get('method').value = 'ma'; get('method').onchange();
  assert.equal(get('tradeoff').children.filter(e => e.tag === 'circle').length,
    data.candidates.filter(c => c.config.method === 'ma').length);
  get('vertical').value = 'score'; get('vertical').onchange();
  for (const id of ['tradeoff', 'timeline', 'heatmap', 'validation']) {
    for (const e of descendants(get(id))) for (const value of Object.values(e.attributes)) {
      assert.ok(!/NaN|Infinity|undefined/.test(value), `${id}: ${value}`);
    }
  }
});

test('scatter clicks and table buttons select a trace configuration', () => {
  const point = get('tradeoff').children.find(e => e.tag === 'circle');
  point.events.click();
  assert.equal(get('traceConfig').value, 'chosen');
  assert.ok(get('selection').textContent.includes('移动平均'));
  assert.ok(get('traceSummary').textContent.includes('移动平均'));
  const button = descendants(get('candidateRows')) .find(e => e.tag === 'button');
  button.onclick();
  assert.ok(get('selection').textContent.includes(String(data.candidates[0].config.threshold)));
});

test('every held-out session can be plotted and fixed baseline can be selected', () => {
  for (let i = 0; i < data.traces.length; i++) {
    get('session').value = String(i); get('traceConfig').value = 'heldout';
    get('session').onchange();
    assert.ok(get('traceSummary').textContent.includes(data.traces[i].id));
    assert.ok(get('traceSummary').textContent.includes('本轮不参与参数选择'));
  }
  get('traceConfig').value = 'baseline'; get('traceConfig').onchange();
  assert.ok(get('traceSummary').textContent.includes('不平滑 / 阈值 50'));
});

test('browser filter uses the same hand-checked causal recurrence', () => {
  const ema = vm.runInContext('filtered([{attention:20},{attention:80},{attention:80}],{method:"ema",parameter:.5})', context);
  assert.deepEqual(Array.from(ema), [20, 50, 65]);
  const ma = vm.runInContext('filtered([{attention:20},{attention:20},{attention:80},{attention:80}],{method:"ma",parameter:3})', context);
  assert.deepEqual(Array.from(ma), [20, 20, 40, 60]);
});
