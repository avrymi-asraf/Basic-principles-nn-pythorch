
DATA = 0
LEABL = 1
class ModelTrainer:
    def __init__(self,model:Module,data:Tuple[Tensor,Tensor],loss_function:Callable[[Tensor,Tensor],Tensor],optimizer:Callable[[],None],get_parameters:Callable[[],Tuple[float,float]]):
        self.model = model
        self.data = data
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.get_parameters = get_parameters
        self.len_data:int = len(data[DATA])
        self.counter:int = 0
        self.loss_list:List[float] = []
    def step(self,num_steps:int=1):
        for i in range(num_steps):
            input_data = self.data[DATA][self.counter]
            target = self.data[LEABL][self.counter]
            out = self.model(input_data)
            diff = self.loss_function(out, target)
            diff.backward()
            self.loss_list.append(diff.item())
            self.optimizer.step()
            self.counter = (self.counter + 1)%self.len_data # repiat on data
    def get_less(self)->List[float]:  
        return self.loss_list          







