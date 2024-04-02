# import aclick
import aclick
import multiprocessing
from MMSR.model.runner import Runner
from MMSR.model.utils.const_improver import OptimizationType
multiprocessing.set_start_method('spawn', force=True)
print('$'*50)
import os

# 指定使用第二个GPU(从0开始编号)
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from MMSR.model.runner import Runner
from MMSR.model.utils.const_improver import OptimizationType
print('os.environ = 3')
# from model.runner import Runner
# from model.utils.const_improver import OptimizationType

@aclick.command("predict")
def main(
    num_equations: int = 1,
    optimization_type: OptimizationType = "gradient",
):
    # function = '((x) ^ (8))+((x) ^ (7))+((x) ^ (6))+((x) ^ (5))+((x) ^ (4))+((x) ^ (3))+((x) ^ (2))+x'
    function = '((x) ^ (4))'
    print('((x) ^ (4))')
    model = '/home/wangyingli/liyanjie/mutimodal/None/checkpoints/None/'
    print("model name passed to the function...")
    runner = Runner.from_checkpoint(
        model, num_equations=num_equations, optimization_type=optimization_type
    )
    predicted = runner.predict(function)
    print("Function:", predicted[0])
    print("R2:", predicted[1])
    print("Relative error:", predicted[2])


if __name__ == "__main__":

    main()