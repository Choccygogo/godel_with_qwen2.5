import json
import sys
import os
import multiprocessing as mp

if __name__ == '__main__':
    mp.set_start_method('fork')
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(project_root)
    from prover.lean.verifier import Lean4ServerScheduler

    output_path = "./code_compilation.json"
    cpu = 64
    input_file_path = "./copy.json"

    with open(input_file_path, 'r') as json_file:
        codes = json.load(json_file)

    lean4_scheduler = Lean4ServerScheduler(max_concurrent_requests=cpu, timeout=300, memory_limit=10, name='verifier')

    request_id_list = lean4_scheduler.submit_all_request([code["code"] for code in codes])
    outputs_list = lean4_scheduler.get_all_request_outputs(request_id_list)
    lean4_scheduler.close()

    assert len(outputs_list) == len(codes)
    ana_result = []
    for i in range(len(codes)):
        codes[i]["compilation_result"] = outputs_list[i]
        ana_result.append(
            {"name": codes[i]["name"],
             "compilation_result": outputs_list[i]["complete"]}
        )
    with open(output_path, 'w') as json_file:
        json.dump(codes, json_file, indent=4)
