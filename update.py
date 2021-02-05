import json
import hashlib
import os
import sys
import glob

from tvm.te import ComputeOp, PlaceholderOp

from tvm.auto_scheduler import save_records
from tvm.auto_scheduler.measure import MeasureInput
from tvm.auto_scheduler.measure_record import load_records
from tvm.auto_scheduler.utils import get_const_tuple

def update_file(log_file, tasks):
    new_log_file = log_file

    def get_old_hash_key(dag):
        """Return the hash key of a compute DAG."""
        str_key = ""
        for op in dag.ops:
            t = op.output(0)
            if isinstance(op, PlaceholderOp):
                str_key += "placeholder,"
                str_key += str(get_const_tuple(t.shape)) + ","
                str_key += t.dtype + ";"
            elif isinstance(op, ComputeOp):
                str_key += str(t.op.body) + ","
                str_key += str(get_const_tuple(t.shape)) + ","
                str_key += t.dtype + ";"
            else:
                raise ValueError("Invalid op: " + op)

        str_key = str_key.encode(encoding="utf-8")
        return hashlib.md5(str_key).hexdigest()

    # Establish the key mapping
    old_key_to_task = {}
    hit_count = {}
    for idx, task in enumerate(tasks):
        old_key = json.dumps((get_old_hash_key(task.compute_dag),))
        old_key_to_task[old_key] = task
        hit_count[old_key] = 0
        print("Task %d %s -> %s" % (idx, old_key, task.workload_key))

    # Update the workload key in an existing log file
    new_inputs = []
    new_results = []
    for inp, res in load_records(log_file):
        if inp.task.workload_key not in old_key_to_task:
            print(
                "Ignore key %s in log file due to no corresponding task found" % inp.task.workload_key
            )
            continue
        hit_count[inp.task.workload_key] += 1
        new_inputs.append(MeasureInput(old_key_to_task[inp.task.workload_key], inp.state))
        new_results.append(res)

    for key, cnt in hit_count.items():
        print("Old key %s hits %d times" % (key, cnt))

    if os.path.exists(new_log_file):
        os.remove(new_log_file)
    save_records(new_log_file, new_inputs, new_results)

