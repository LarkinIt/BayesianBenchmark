import petab
import roadrunner

class ModelProblem:
    r"""
    The model problem class that contains the libroadrunner simulator object and PEtab information. It is directly based on pyPESTO roadrunner importer (https://github.com/ICB-DCM/pyPESTO/tree/main/pypesto/objective/roadrunner) 
    
    Parameters
    ----------
    model_name : str
        The local directory containing the PEtab files 
    """
    def __init__(self, 
                 model_name : str):
       config_file = f"{model_name}/{model_name}.yaml"
       # load the PEtab problem
       self.petab_problem = petab.Problem.from_yaml(config_file)
       
       # verify PEtab problem is properly formatted
       if petab.lint_problem(self.petab_problem):
           raise ValueError("Something is wrong with the PEtab model >.<")
       
       # load the libroadrunner model
       model_file = f"{model_name}/model_{self.problem.model.model_id}.xml"
       self.simulator = roadrunner.RoadRunner(model_file)
       
       
       