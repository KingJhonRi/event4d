from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from event4d.query.engine import parse_query,QueryEngine,LanguageAnswerer

ap=argparse.ArgumentParser(); ap.add_argument("memory",type=Path); ap.add_argument("--examples",action="store_true"); args=ap.parse_args()
with args.memory.open() as f:
    rows=[json.loads(line) for line in f if line.strip()]
if not rows: raise ValueError("memory log is empty")
engine=QueryEngine(rows[-1]); answerer=LanguageAnswerer()
examples=["What objects are near me?","What is the closest object?","What objects are approaching me?","What moved during the last five seconds?","What is within 40 meters?","Where is object 3?"]
if args.examples:
    result=[]
    for q in examples:
        result.append(f"> {q}\n{answerer.answer(parse_query(q),engine.execute(parse_query(q)))}")
    text="\n\n".join(result)+"\n"
    print(text)
    (args.memory.parent/"query_examples.txt").write_text(text,encoding="utf-8")
else:
    print("4D Memory Query (type exit to quit)")
    while True:
        try: q=input("> ").strip()
        except EOFError: break
        if q.lower() in ("exit","quit"): break
        spec=parse_query(q)
        print(answerer.answer(spec,engine.execute(spec)))
