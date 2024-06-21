import petab
import roadrunner

class ModelProblem:
    r"""
    The model problem class that contains the libroadrunner simulator object and PEtab information. 
    
    Parameters
    ----------
    model_name : str
        The local directory containing the PEtab files 
    """
    def __init__(self, 
                 model_name : str):
       config_file = f"{model_name}/{model_name}.yaml"
       # load the PEtab problem
       self.problem = petab.Problem.from_yaml(config_file)
       
       # load the libroadrunner model
       model_file = f"{model_name}/model_{self.problem.model.model_id}.xml"
       self.simulator = roadrunner.RoadRunner(model_file)
       