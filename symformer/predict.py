import aclick
import multiprocessing
from symformer.model.runner import Runner
from symformer.model.utils.const_improver import OptimizationType
multiprocessing.set_start_method('spawn', force=True)
print('$'*50)
@aclick.command("predict")
def main(
    function: str,
    /,
    model: str,
    num_equations: int = 1,
    optimization_type: OptimizationType = "gradient",
):
    runner = Runner.from_checkpoint(
        model, num_equations=num_equations, optimization_type=optimization_type
    )
    function = 'sin(x) + y'
    predicted = runner.predict(function)
    print("Function:", predicted[0])
    print("R2:", predicted[1])
    print("Relative error:", predicted[2])

if __name__ == "__main__":
    main()
