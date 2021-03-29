from mmcv.runner import HOOKS, master_only
from mmcv.runner.hooks import LoggerHook


@HOOKS.register_module()
class CometMLLoggerHook(LoggerHook):

    def __init__(self,
                 api_key=None,
                 project_name=None,
                 hyper_params=None,
                 import_comet=False,
                 interval=10,
                 ignore_last=True,
                 reset_flag=True,
                 by_epoch=True):
        """Class to log metrics to Comet ML.
        It requires `comet_ml` to be installed.
        Args:
            api_key (str, optional): Your API key obtained from comet.ml
            project_name (str, optional):
                Send your experiment to a specific project. 
                Otherwise will be sent to Uncategorized Experiments. 
                If project name does not already exists Comet.ml will create 
                a new project.
            hyper_params (dict, optional): Logs a dictionary 
                (or dictionary-like object) of multiple parameters.
            import_comet (bool optional): Whether to import comet_ml before run.
                WARNING: Comet ML have to be imported before sklearn and torch,
                or COMET_DISABLE_AUTO_LOGGING have to be set in the environment.
            interval (int): Logging interval (every k iterations).
            ignore_last (bool): Ignore the log of last iterations in each epoch
                if less than `interval`.
            reset_flag (bool): Whether to clear the output buffer after logging
            by_epoch (bool): Whether EpochBasedRunner is used.
        """
        super(CometMLLoggerHook, self).__init__(interval, ignore_last,
                                                reset_flag, by_epoch)
        if import_comet:
            self.import_comet()
        self.api_key = api_key
        self.project_name = project_name
        self.hyper_params = hyper_params

    def import_comet(self):
        try:
            import comet_ml
        except ImportError:
            raise ImportError(
                'Please run "pip install comet_ml" to install Comet ML')
        self.comet_ml = comet_ml

    @master_only
    def before_run(self, runner):
        self.experiment = comet_ml.Experiment(
            api_key=self.api_key,
            project_name=self.project_name,
        )
        if self.hyper_params is not None:
            self.experiment.log_parameters(self.hyper_params)

    @master_only
    def log(self, runner):
        tags = self.get_loggable_tags(runner, allow_text=True)
        for tag, val in tags.items():
            self.experiment.log_metric(name=tag,
                                       value=val,
                                       step=self.get_iter(runner),
                                       epoch=self.get_epoch(runner))

    @master_only
    def after_run(self, runner):
        self.experiment.end()
