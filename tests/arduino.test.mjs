import { execFileSync } from 'node:child_process';
import assert from 'node:assert/strict';
const packet = (p) => [170,170,p.length,...p,(~p.reduce((a,b)=>a+b,0))&255];
const powers = [0x18d48b,0x13d169,0,0xffffff,0x008000,0x800000,0x123456,1];
const payload = [2,0,131,24,...powers.flatMap(n=>[n>>>16,(n>>>8)&255,n&255]),4,70,5,42];
function check(bytes, expected) {
  const text = execFileSync(process.argv[2], {input:bytes.join(' '),encoding:'utf8',timeout:3000});
  assert.equal(text.replaceAll('\r\n','\n'),expected);
}
const row = [...powers,70].join(',')+'\n';
check(packet(payload),row);
check([...packet(payload),...packet(payload)],row+row);
const corrupt = packet(payload); corrupt[corrupt.length-1] ^= 1;
check(corrupt,'');
check(packet(payload.map((v,i)=>i===1?200:v)),'');
check([...packet([128,2,170,170]),...packet(payload)],row);
check(packet(payload.map((v,i)=>i===2?130:v)),'');
console.log('PASS: decoding, CSV rows, checksum, signal quality, small packets, field markers');
