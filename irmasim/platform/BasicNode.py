from irmasim.platform.TaskRunner import TaskRunner


class BasicNode(TaskRunner):

    def __init__(self, id: str, config: dict):
        super(BasicNode, self).__init__(id=id, config=config)
