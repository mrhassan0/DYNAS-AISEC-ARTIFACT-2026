import copy
import os
import yaml
import os.path as osp


SCENARIO_DIR = osp.dirname(osp.abspath(__file__))

# Scenario loaders mutate the dictionary returned by ``load_yaml`` while they
# expand ranged subnet sizes and normalize the remaining fields.  Keep one
# pristine parsed template per file and hand each loader an independent copy.
# Besides avoiding cross-load state leaks, this prevents large dynamic
# scenarios from being reparsed for every episode in every worker process.
_YAML_CACHE = {}

# default subnet address for internet
INTERNET = 0

# Constants
NUM_ACCESS_LEVELS = 2
NO_ACCESS = 0
USER_ACCESS = 1
ROOT_ACCESS = 2
DEFAULT_HOST_VALUE = 0

# scenario property keys
SUBNETS = "subnets"
TOPOLOGY = "topology"
SENSITIVE_HOSTS = "sensitive_hosts"
SERVICES = "services"
OS = "os"
PROCESSES = "processes"
EXPLOITS = "exploits"
PRIVESCS = "privilege_escalation"
SERVICE_SCAN_COST = "service_scan_cost"
OS_SCAN_COST = "os_scan_cost"
SUBNET_SCAN_COST = "subnet_scan_cost"
PROCESS_SCAN_COST = "process_scan_cost"
HOST_CONFIGS = "host_configurations"
FIREWALL = "firewall"
HOSTS = "host"
STEP_LIMIT = "step_limit"
ACCESS_LEVELS = "access_levels"

# scenario exploit keys
EXPLOIT_SERVICE = "service"
EXPLOIT_OS = "os"
EXPLOIT_PROB = "prob"
EXPLOIT_COST = "cost"
EXPLOIT_ACCESS = "access"

# scenario privilege escalation keys
PRIVESC_PROCESS = "process"
PRIVESC_OS = "os"
PRIVESC_PROB = "prob"
PRIVESC_COST = "cost"
PRIVESC_ACCESS = "access"

# host configuration keys
HOST_SERVICES = "services"
HOST_PROCESSES = "processes"
HOST_OS = "os"
HOST_FIREWALL = "firewall"
HOST_VALUE = "value"


def load_yaml(file_path):
    """Load yaml file located at file path.

    Parameters
    ----------
    file_path : str
        path to yaml file

    Returns
    -------
    dict
        contents of yaml file

    Raises
    ------
    Exception
        if theres an issue loading file. """
    canonical_path = osp.realpath(osp.abspath(osp.expanduser(os.fspath(file_path))))
    stat = os.stat(canonical_path)
    signature = (stat.st_mtime_ns, stat.st_size)
    cached = _YAML_CACHE.get(canonical_path)

    if cached is None or cached[0] != signature:
        with open(canonical_path, encoding="utf-8") as fin:
            content = yaml.load(fin, Loader=yaml.FullLoader)
        _YAML_CACHE[canonical_path] = (signature, content)
        cached = _YAML_CACHE[canonical_path]

    return copy.deepcopy(cached[1])


def get_file_name(file_path):
    """Extracts the file or dir name from file path

    Parameters
    ----------
    file_path : str
        file path

    Returns
    -------
    str
        file name with any path and extensions removed
    """
    full_file_name = file_path.split(os.sep)[-1]
    file_name = full_file_name.split(".")[0]
    return file_name
