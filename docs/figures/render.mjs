// Render docs/figures/*.mmd with pele (https://pele.run) to docs/images/<name>-light.svg and -dark.svg,
// coloured like GitHub's light and dark themes. Run from the repository root:
//     npm install pele && node docs/figures/render.mjs
import { render } from 'pele';
import { readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const THEMES = {
  light: { bg: '#ffffff', fg: '#1f2328', muted: '#59636e', line: '#59636e', surface: '#f6f8fa', 'surface-alt': '#eaeef2',
           border: '#d1d9e0', accent: '#0969da', 'series-1': '#0969da', 'series-2': '#bf8700', 'series-3': '#8c959f' },
  dark: { bg: '#0d1117', fg: '#f0f6fc', muted: '#9198a1', line: '#9198a1', surface: '#151b23', 'surface-alt': '#212830',
          border: '#3d444d', accent: '#4493f8', 'series-1': '#4493f8', 'series-2': '#d29922', 'series-3': '#656c76' },
};
const FONT = '-apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans", Helvetica, Arial, sans-serif';
const dir = 'docs/figures', out = 'docs/images';
for (const f of readdirSync(dir).filter((f) => f.endsWith('.mmd'))) {
  const name = f.replace(/\.mmd$/, '');
  for (const [theme, vars] of Object.entries(THEMES)) {
    const { svg } = render(readFileSync(join(dir, f), 'utf8'), { fontFamily: FONT, idPrefix: `${name}-` });
    const css = Object.entries(vars).map(([k, v]) => `--pele-${k}:${v};`).join('') + `--pele-font:${FONT.replaceAll('"', "'")};`;
    writeFileSync(join(out, `${name}-${theme}.svg`), svg.replace('style="', `style="${css}`));
  }
  console.log(name);
}
