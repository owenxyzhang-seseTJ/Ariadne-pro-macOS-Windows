import {build, context} from 'esbuild';
import {cp, mkdir} from 'node:fs/promises';
await mkdir('dist/assets',{recursive:true});
await cp('public','dist',{recursive:true});
await cp('index.html','dist/index.html');
const options={entryPoints:['src/main.tsx'],bundle:true,format:'esm',target:['es2022'],outfile:'dist/assets/app.js',minify:true,sourcemap:true,logLevel:'info'};
if(process.argv.includes('--watch')){const c=await context(options);await c.watch();console.log('Watching frontend source.');}else await build(options);
