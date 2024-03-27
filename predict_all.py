# import aclick

from symformer.model.runner import Runner
from symformer.model.utils.const_improver import OptimizationType
import multiprocessing
import pandas as pd
import re
import numpy as np

# from model.runner import Runner
# from model.utils.const_improver import OptimizationType
multiprocessing.set_start_method('spawn', force=True)
print('$'*50)
import os

# 指定使用第二个GPU(从0开始编号)
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
from symformer.model.runner import Runner
from symformer.model.utils.const_improver import OptimizationType
print('os.environ = 2')

def process_benchmark(expression):
    pow_regexp = r"pow\((.*?),(.*?)\)"
    pow_replace = r"((\1) ^ (\2))"
    processed = re.sub(pow_regexp, pow_replace, expression)

    div_regexp = r"div\((.*?),(.*?)\)"
    div_replace = r"((\1) / (\2))"
    processed = re.sub(div_regexp, div_replace, processed)
    processed = processed.replace("x1", "x")
    processed = processed.replace("x2", "y")
    return processed


# @aclick.command("predict")
def main(
        num_equations: int = 1,
        optimization_type: OptimizationType = "gradient",
):
    # function = '2*cos(x**2+y) + 4.5325'
    benchmarks_path = './symformer/assets/benchmarks.csv'
    benchmarks = pd.read_csv(benchmarks_path, header=0)
    benchmarks = benchmarks.to_numpy()
    # print("benchmarks:", benchmarks)

    # function = 'sin(x)+sin(pow(y,2))'
    model = '/home/wangyingli/liyanjie/mutimodal/None/checkpoints/None/'
    print("model name passed to the function...")
    runner = Runner.from_checkpoint(
        model, num_equations=num_equations, optimization_type=optimization_type
    )
    # predicted = runner.predict(function)
    # print("Function:", predicted[0])
    # print("R2:", predicted[1])
    # print("Relative error:", predicted[2])
    results = []
    best_results = []
    for line in benchmarks:
        all_r2 = []
        all_re = []

        print("line:", line)
        if line[1] == 2 or line[1] == 1:
            print("line[2]:", line[2])
            for i in range(10):
                expr = process_benchmark(line[2])
                try:
                    predicted = runner.predict(expr)
                    pred_fun = predicted[0]
                    r2 = predicted[1]
                    re = predicted[2]
                    results.append([line[0], expr, pred_fun, r2, re])
                    df = pd.DataFrame(results, columns=['expr', 'true_expr', 'pred_expr', 'r2', 're'])
                    df.to_csv('./symformer_all_1.csv')
                    all_r2.append(r2)
                    all_re.append(re)
                    print("all_r2:", all_r2)
                except:
                    i -= 1
            best_results.append([line[0], np.max(np.array(all_r2)), np.min(np.array(all_re))])
            df_best = pd.DataFrame(best_results, columns=['expr', 'r2', 're'])
            df_best.to_csv('./symformer_d2_best.csv')


if __name__ == "__main__":
    main()