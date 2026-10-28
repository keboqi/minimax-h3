// Small file protocol around the official SDK; all inference stays in the native runtime.
import fs from 'node:fs';
import {pathToFileURL} from 'node:url';

const job = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const {Align} = await import(pathToFileURL(job.module).href);
const align = await Align.load({cacheRoot: job.cache});
const audio = fs.openSync(job.audio, 'r');
const duration = fs.fstatSync(audio).size / (16000 * 4);
const results = [];
try {
  for (const group of job.groups) {
    const words = group.words ?? [];
    if (!words.length || !Align.isSupported(group.language_code)) {
      results.push({error: group.error || 'No acoustic word timing or unsupported language'});
      continue;
    }
    try {
      const start = Math.max(0, Math.floor((words[0].start - 1.5) * 16000));
      const end = Math.min(Math.floor(duration * 16000), Math.ceil((words.at(-1).end + 1.5) * 16000));
      if (end <= start) throw new Error('Invalid audio window');
      const bytes = Buffer.alloc((end - start) * 4);
      const count = fs.readSync(audio, bytes, 0, bytes.length, start * 4);
      if (count !== bytes.length) throw new Error('Incomplete audio window');
      const samples = new Float32Array(end - start);
      for (let i = 0; i < samples.length; i++) samples[i] = bytes.readFloatLE(i * 4);
      const offset = start / 16000;
      const local = words.map(w => ({...w, start: w.start - offset, end: w.end - offset}));
      const refined = await align.refine(samples, 16000, local, {language: group.language_code});
      results.push({words: refined.map(w => ({...w, start: w.start + offset, end: w.end + offset}))});
    } catch (error) {
      results.push({error: String(error.message || error)});
    }
  }
  fs.writeFileSync(job.output, JSON.stringify(results));
} finally {
  fs.closeSync(audio);
  align.dispose();
}
