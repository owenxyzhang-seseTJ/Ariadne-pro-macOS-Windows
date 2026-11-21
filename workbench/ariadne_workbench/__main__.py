import argparse
import uvicorn

parser=argparse.ArgumentParser(description='Ariadne Pro 本地工作台')
parser.add_argument('--port',type=int,default=8790)
args=parser.parse_args()
uvicorn.run('ariadne_workbench.app:app',host='127.0.0.1',port=args.port,log_level='info')
