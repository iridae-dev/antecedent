import hashlib,json,re,subprocess
from pathlib import Path
root=Path('/Users/charleshinshaw/Desktop/antecedent')
archive=root/'target/calibration-replay/recordless-count-recovery-767659966'
planfile=archive/'plan.json';plan=json.loads(planfile.read_text(encoding='utf-8'))
assert subprocess.check_output(['git','-C',plan['worktree'],'rev-parse','HEAD'],text=True).strip()==plan['source']
assert not subprocess.check_output(['git','-C',plan['worktree'],'status','--porcelain'],text=True).strip()
results=[]
for case in plan['cases']:
 raw=archive/(case['package']+'.log');text=raw.read_text(encoding='utf-8')
 lines=[x.split('calibration-assertion-receipt ',1)[1] for x in text.splitlines() if 'calibration-assertion-receipt ' in x]
 assert len(lines)==1;receipt=json.loads(lines[0]);assert receipt['profile']==case['profile'];assert receipt['attempted']==case['attempted'];assert receipt['rows']==case['rows'];assert receipt['reported']<=receipt['attempted']
 match=re.search(r'test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; \d+ filtered out; finished in ([0-9.]+)s',text);assert match
 result={'schema':'original_recordless_count_recovery_receipt.v1','source':plan['source'],'original_source':plan['original_source'],'plan_sha256':hashlib.sha256(planfile.read_bytes()).hexdigest(),'group':case['group'],'test':case['test'],'binary_sha256':case['binary_sha256'],'raw_log':str(raw.relative_to(root)),'raw_log_sha256':hashlib.sha256(raw.read_bytes()).hexdigest(),'rust_harness_passed':True,'rust_harness_elapsed_seconds':float(match[1]),'process_exit_code_observed':0 if case['package']=='antecedent-stats' else None,'receipt':receipt,'new_calibration_claim':False,'postprocessing_note':'Initial controller required the marker at line start; libtest printed its test prefix on the same line. Recovered only existing unchanged raw JSON bytes after both Rust harnesses completed with one passed and zero failed; no scientific rerun. The controller directly observed stats exit zero, then failed postprocessing before waiting for the independently completed sensitivity child; no sensitivity OS exit code is fabricated.'}
 (archive/(case['package']+'.receipt.json')).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');results.append(result)
(archive/'recovery-summary.json').write_text(json.dumps({'source':plan['source'],'results':results,'initial_controller_log':'target/calibration-audit/original-recordless-count-recovery-execution.log'},indent=2)+'\n',encoding='utf-8')
print(json.dumps([(x['group'],x['receipt']) for x in results],indent=2))
