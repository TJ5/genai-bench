import json
import os
import re
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

from genai_bench.logging import init_logger
from genai_bench.metrics.metrics import AggregatedMetrics, RequestLevelMetrics
from genai_bench.protocol import ExperimentMetadata

logger = init_logger(__name__)

MetricsData = (
    Dict[Literal["aggregated_metrics"], AggregatedMetrics]
    | Dict[Literal["individual_metrics"], List[RequestLevelMetrics]]
)

ExperimentMetrics = Dict[
    str,  # traffic-scenario
    Dict[
        int,  # concurrency-level
        MetricsData,
    ],
]


def load_multiple_experiments(
    folder_name: str, filter_criteria=None
) -> List[Tuple[ExperimentMetadata, ExperimentMetrics]]:
    """
    Loads the JSON files from one experiment folder. The folder should contain
    a list of subfolders, each subfolder corresponding to an experiment.

    Args:
        folder_name (str): Path to the folder containing the experiment data.
        filter_criteria (dict, optional): Dictionary of filtering criteria based
            on metadata keys.

    Returns:
        list: A list of tuples (run_data, experiment_metadata) from each
            subfolder.
    """
    run_data_list = []

    # Loop through subfolders and files in the folder
    for subfolder in os.listdir(folder_name):
        subfolder_path = os.path.join(folder_name, subfolder)
        if os.path.isdir(subfolder_path):
            # Recursively load data from subfolders
            metadata, run_data = load_one_experiment(subfolder_path, filter_criteria)
            if metadata and run_data:
                run_data_list.append((metadata, run_data))

    return run_data_list


def load_one_experiment(
    folder_name: str, filter_criteria: Optional[Dict[str, Any]] = None
) -> Tuple[Optional[ExperimentMetadata], ExperimentMetrics]:
    """
    Processes files in the provided folder (for metadata and run data).

    Args:
        folder_name (str): Path to the folder.
        filter_criteria (dict, optional): Dictionary of filtering criteria.

    Returns:
        ExperimentMetadata: ExperimentMetadata object.
        dict: Dictionary containing the metrics for each scenario and
            concurrency.
    """
    experiment_file = os.path.join(folder_name, "experiment_metadata.json")

    if not os.path.exists(experiment_file):
        return None, {}

    experiment_metadata = load_experiment_metadata(experiment_file, filter_criteria)
    if not experiment_metadata:
        return None, {}

    run_data: ExperimentMetrics = {}

    for file_name in sorted(os.listdir(folder_name)):
        file_path = os.path.join(folder_name, file_name)
        if re.match(
            r"^.+_.+_(?:concurrency|batch_size|request_rate)_\d+_time_\d+s\.json$",
            file_name,
        ):
            load_run_data(file_path, run_data, filter_criteria)

    if not run_data:
        return experiment_metadata, run_data

    for scenario in experiment_metadata.traffic_scenario:
        if scenario not in run_data:
            logger.warning(
                f"‼️ Scenario {scenario} in metadata but metrics not found! "
                f"Please re-run this scenario if necessary!"
            )
            experiment_metadata.traffic_scenario.remove(scenario)

    # Determine which iteration types to check for this experiment
    # Always check the primary iteration_type for backward compatibility
    # Additionally check request_rate if present (for mixed experiments)
    iteration_types_present = [experiment_metadata.iteration_type]

    # Add request_rate if present (for mixed experiments with both
    # request_rate and num_concurrency)
    if experiment_metadata.request_rate:
        iteration_types_present.append("request_rate")

    # Build expected values from all present iteration types
    expected_values_map = {
        "batch_size": experiment_metadata.batch_size or [],
        "num_concurrency": experiment_metadata.num_concurrency,
        "request_rate": experiment_metadata.request_rate or [],
    }
    expected_concurrency = set()
    for it_type in iteration_types_present:
        expected_concurrency.update(expected_values_map.get(it_type, []))

    # Check if any scenarios are missing levels for any iteration type
    for scenario_key, scenario_data in run_data.items():
        # Collect seen values from all relevant levels keys
        # Note: scenario_data may contain string keys like "_levels" for metadata
        seen_values: Set[int] = set()
        for it_type in iteration_types_present:
            levels_key = f"{it_type}_levels"
            seen_for_type = scenario_data.get(levels_key)  # type: ignore[call-overload]
            if isinstance(seen_for_type, set):
                seen_values.update(seen_for_type)

        missing_concurrency: List[Any] = sorted(expected_concurrency - seen_values)
        if missing_concurrency:
            # Build a descriptive message about which types are missing
            missing_by_type = {}
            for it_type in iteration_types_present:
                levels_key = f"{it_type}_levels"
                seen_for_type = scenario_data.get(levels_key)  # type: ignore[call-overload]
                if not isinstance(seen_for_type, set):
                    seen_for_type = set()
                expected_for_type = set(expected_values_map.get(it_type, []))
                missing_for_type = sorted(expected_for_type - seen_for_type)
                if missing_for_type:
                    missing_by_type[it_type] = missing_for_type

            if missing_by_type:
                # Use old format for single-type experiments (backward compatibility)
                # Use new format for mixed experiments
                if len(missing_by_type) == 1:
                    it_type, missing_values = next(iter(missing_by_type.items()))
                    logger.warning(
                        f"‼️ Scenario '{scenario_key}' is missing {it_type} "
                        f"levels: {missing_values}. "
                        f"Please re-run this scenario if necessary!"
                    )
                else:
                    missing_desc = ", ".join(
                        [f"{k}: {v}" for k, v in missing_by_type.items()]
                    )
                    logger.warning(
                        f"‼️ Scenario '{scenario_key}' is missing levels: "
                        f"{missing_desc}. Please re-run this scenario if necessary!"
                    )

        # Remove ALL _levels metadata keys (not just the one matching
        # iteration_type). This is necessary because mixed experiments can have
        # both num_concurrency_levels and request_rate_levels keys, and we need
        # to remove all of them
        keys_to_remove = [
            k for k in scenario_data if isinstance(k, str) and k.endswith("_levels")
        ]
        for key in keys_to_remove:
            scenario_data.pop(key, None)  # type: ignore[arg-type]

    return experiment_metadata, run_data


def load_experiment_metadata(
    file_path: str, filter_criteria: Optional[Dict[str, Any]] = None
) -> Optional[ExperimentMetadata]:
    """
    Loads the experiment metadata from the provided JSON file path.

    Args:
        file_path (str): Path to the `experiment_metadata.json` file.
        filter_criteria (dict, optional): Dictionary of filtering criteria.

    Returns:
        ExperimentMetadata: Filtered ExperimentMetadata object.
    """
    with open(file_path, "r") as file:
        data = json.load(file)
        experiment_metadata = ExperimentMetadata(**data)

    if filter_criteria and not apply_filter_to_metadata(
        experiment_metadata, filter_criteria
    ):
        logger.info(
            f"No match with filter_criteria found in ExperimentMetadata under "
            f"{file_path}."
        )
        return None  # Metadata does not match the filter

    return experiment_metadata


def apply_filter_to_metadata(
    experiment_metadata: ExperimentMetadata, filter_criteria: Dict[str, Any]
) -> bool:
    """
    Applies filter criteria to the experiment metadata.

    Args:
        experiment_metadata (ExperimentMetadata): The ExperimentMetadata object
            to filter.
        filter_criteria (dict): The dictionary of filter keys and values.

    Returns:
        bool: True if the metadata matches the filter, False otherwise.
    """
    for key, val in filter_criteria.items():
        if key not in experiment_metadata.model_fields:
            logger.info(f"Filter key {key} is not in the metadata.")
            return False  # Key not present

        if key == "traffic_scenario":
            if not isinstance(val, list):
                val = [val]
            filtered_scenarios = set(experiment_metadata.traffic_scenario).intersection(
                set(val)
            )
            experiment_metadata.traffic_scenario = list(filtered_scenarios)
            if not filtered_scenarios:
                logger.info(
                    f"The scenarios {val} you want to filter is not "
                    f"presented in your experiments."
                )
                return False  # No matching scenarios
        elif getattr(experiment_metadata, key) != val:
            logger.info(
                f"Filter {key}:{val} does not match the value in "
                f"experiment metadata: {getattr(experiment_metadata, key)}"
            )
            return False  # Metadata value doesn't match

    return True


def load_run_data(
    file_path: str,
    run_data: dict,
    filter_criteria: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Loads run data from individual scenario JSON files and filters based on
    criteria.

    Args:
        file_path (str): Path to the JSON file containing metrics.
        run_data (dict): Dictionary where the data will be stored.
        filter_criteria (dict, optional): Filtering criteria.
    """
    with open(file_path, "r") as file:
        data = json.load(file)
        aggregated_metrics = AggregatedMetrics(**data["aggregated_metrics"])
        scenario = aggregated_metrics.scenario

        if (
            filter_criteria
            and "traffic_scenario" in filter_criteria
            and scenario not in filter_criteria["traffic_scenario"]
        ):
            return  # Skip if scenario not in the filtered list

        # Get the iteration type and value
        iteration_type = aggregated_metrics.iteration_type
        if iteration_type == "batch_size":
            iteration_value = aggregated_metrics.batch_size
        elif iteration_type == "request_rate":
            iteration_value = aggregated_metrics.request_rate
            if iteration_value is None:
                # Skip if request_rate is None (shouldn't happen for request_rate runs)
                return
        else:
            iteration_value = aggregated_metrics.num_concurrency

        # Store iteration values in scenario data
        iteration_key = f"{iteration_type}_levels"
        run_data.setdefault(scenario, {}).setdefault(iteration_key, set()).add(
            iteration_value
        )

        # Store the metrics data
        run_data.setdefault(scenario, {})[iteration_value] = {
            "aggregated_metrics": aggregated_metrics,
            "individual_request_metrics": data.get("individual_request_metrics", []),
        }
