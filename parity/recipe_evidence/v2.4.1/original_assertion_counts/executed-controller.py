import hashlib,json,os,re,subprocess,time
from pathlib import Path
root=Path('/Users/charleshinshaw/Desktop/antecedent')
planfile=root/'target/calibration-audit/original-recordless-count-recovery-plan.json'
plan=json.loads(planfile.read_text(encoding='utf-8'))
wt=Path(plan['worktree'])
def git(*args):return subprocess.check_output(['git','-C',str(wt),*args],text=True).strip()
assert git('rev-parse','HEAD')==plan['source']
assert not git('status','--porcelain')
log=root/'target/calibration-audit/final-attestation-count-witness-prebuild.log'
bins=re.findall(r'Executable unittests src/lib.rs \((.*?)\)',log.read_text(encoding='utf-8'))
assert len(bins)==2,bins
archive=root/'target/calibration-replay'/('recordless-count-recovery-'+plan['source'][:9]);archive.mkdir(parents=True,exist_ok=False)
env={k:v for k,v in os.environ.items() if not k.startswith('ANTECEDENT_CALIBRATION_')}
env['RUST_TEST_THREADS']='1'
for case in plan['cases']:
 binary=next(Path(b) for b in bins if case['package'].replace('-','_')+'-' in b)
 names=subprocess.check_output([str(binary),'--list','--ignored'],cwd=wt,env=env,text=True)
 exact=[line.removesuffix(': test') for line in names.splitlines() if case['filter'] in line and line.endswith(': test')]
 assert len(exact)==1,exact
 case['binary']=str(binary);case['binary_sha256']=hashlib.sha256(binary.read_bytes()).hexdigest()
 case['command']=[str(binary),exact[0],'--exact','--ignored','--nocapture','--test-threads=1']
 case['test']=exact[0]
plan['prebuild_log_sha256']=hashlib.sha256(log.read_bytes()).hexdigest()
plan['environment_rule']='All ANTECEDENT_CALIBRATION_* overrides removed; original hardcoded data, counts, seeds and execution contexts retained; one test per harness.'
sealed=archive/'plan.json';sealed.write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8')
(archive/'plan.sha256').write_text(hashlib.sha256(sealed.read_bytes()).hexdigest()+'\n',encoding='utf-8')
procs=[]
for case in plan['cases']:
 raw=archive/(case['package']+'.log');handle=raw.open('wb')
 p=subprocess.Popen(case['command'],cwd=wt,env=env,stdout=handle,stderr=subprocess.STDOUT)
 procs.append((p,handle,raw,case,time.time()))
 print('START',case['package'],'pid',p.pid,'context_workers',case['context_workers'],flush=True)
for p,handle,raw,case,start in procs:
 code=p.wait();handle.close();text=raw.read_text(encoding='utf-8')
 matches=[json.loads(x.split('calibration-assertion-receipt ',1)[1]) for x in text.splitlines() if x.startswith('calibration-assertion-receipt ')]
 assert code==0,(case['package'],code,text[-3000:])
 assert len(matches)==1,matches
 receipt=matches[0];assert receipt['profile']==case['profile'];assert receipt['attempted']==case['attempted'];assert receipt['rows']==case['rows']
 result={'source':plan['source'],'original_source':plan['original_source'],'plan_sha256':hashlib.sha256(sealed.read_bytes()).hexdigest(),'group':case['group'],'test':case['test'],'exit_code':code,'raw_log':str(raw.relative_to(root)),'raw_log_sha256':hashlib.sha256(raw.read_bytes()).hexdigest(),'elapsed_seconds':time.time()-start,'receipt':receipt,'new_calibration_claim':False}
 (archive/(case['package']+'.receipt.json')).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
 print('DONE',case['package'],json.dumps(receipt),flush=True)
assert git('rev-parse','HEAD')==plan['source'];assert not git('status','--porcelain')
print('ALL_DONE',archive,flush=True)
