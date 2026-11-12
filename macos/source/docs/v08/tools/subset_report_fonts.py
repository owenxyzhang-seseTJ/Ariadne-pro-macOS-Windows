"""Subset open Noto SC fonts for this report, retaining the OFL notices.

Use --source-dir with the four static NotoSerifSC/NotoSansSC Regular/Bold TTF
files. The distributed source bundle already contains generated subsets.
"""
from pathlib import Path
import argparse
import json
from fontTools import subset
from fontTools.ttLib import TTFont


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-dir', required=True, type=Path)
    args = parser.parse_args()
    report = Path(__file__).resolve().parents[1]
    text = ''.join(p.read_text(encoding='utf-8') for p in report.rglob('*')
                   if p.suffix in {'.tex','.py','.md','.json'} and p.is_file())
    # TeX can synthesize punctuation and accented Latin letters from macros;
    # retain those code points even when they do not occur literally in sources.
    text += ''.join(chr(i) for i in range(32,592))
    text += ''.join(chr(i) for i in range(0x2000,0x2070))
    # Matplotlib logarithmic tick labels use the Unicode mathematical minus.
    text += '\u2212'
    text += '0123456789负原始温度波矢分量参考倒易'
    mapping = {f'Noto{family}SC-{weight}.ttf':f'Report{family}-{weight}.ttf'
               for family in ['Serif','Sans'] for weight in ['Regular','Bold']}
    records=[]
    for original, target in mapping.items():
        source = args.source_dir/original
        font = TTFont(source)
        options = subset.Options()
        options.name_IDs = ['*']
        sub = subset.Subsetter(options=options)
        sub.populate(text=text)
        sub.subset(font)
        family='Ariadne Report Serif' if 'Serif' in original else 'Ariadne Report Sans'
        style='Bold' if 'Bold' in original else 'Regular'
        for name in font['name'].names:
            replacement = {1:family,2:style,4:family+' '+style,6:family.replace(' ','')+'-'+style,
                           16:family,17:style}.get(name.nameID)
            if replacement is not None:
                name.string=replacement.encode(name.getEncoding())
        destination=report/'fonts'/target
        font.save(destination)
        records.append({'file':target,'source_family':original,'bytes':destination.stat().st_size,
                        'modification':'glyph subset and family renamed for report embedding',
                        'license':'SIL Open Font License 1.1; included Noto notices'})
    (report/'fonts'/'PROVENANCE.json').write_text(json.dumps(records,indent=2)+'\n',encoding='utf-8')
    print('Font subsets ready:',sum(r['bytes'] for r in records),'bytes')


if __name__=='__main__':main()
