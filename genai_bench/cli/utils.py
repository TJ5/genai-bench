from locust.env import Environment

import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import gevent

from genai_bench.logging import init_logger

logger = init_logger(__name__)


def manage_run_time(
    max_time_per_run: int,
    max_requests_per_run: int,
    environment: Environment,
) -> int:
    """
    Manages the run time of the benchmarking process by tracking elapsed time
    and ensuring enough requests are completed before the test ends. The
    function will exit when one of the two conditions is met:
    1. The maximum allowed run time (`max_time_per_run`) is reached.
    2. The total number of requests exceeds the maximum requests per run
       (`max_requests_per_run`).

    Args:
        max_time_per_run (int): The maximum allowed run time in seconds.
        max_requests_per_run (int): The maximum number of requests per
            run.
        environment: The environment object with runner stats.

    Returns:
        int: The actual run time in seconds.
    """

    total_run_time = 0

    while total_run_time < max_time_per_run:
        time.sleep(1)
        total_run_time += 1

        assert environment.runner is not None, "environment.runner should not be None"
        total_completed_requests = environment.runner.stats.total.num_requests

        if total_completed_requests >= max_requests_per_run:
            logger.info(
                f"⏩ Exit the run as {total_completed_requests} requests have "
                "been completed."
            )
            break

    return int(total_run_time)


def get_experiment_path(
    experiment_folder_name: Optional[str],
    experiment_base_dir: Optional[str],
    api_backend: str,
    server_engine: Optional[str],
    server_version: Optional[str],
    task: str,
    model: str,
) -> Path:
    """
    Generate experiment path based on provided options and configuration.

    Args:
        experiment_folder_name: Optional custom folder name
        experiment_base_dir: Optional base directory for experiments
            (relative or absolute)
        api_backend: API backend name
        server_engine: Optional server engine name
        server_version: Optional server version
        task: Task name
        model: Model name

    Returns:
        Path: Full path to experiment directory
    """
    # Generate default name if not provided
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_name = (
        f"{api_backend}_"
        f"{server_engine + '_' if server_engine else ''}"
        f"{server_version + '_' if server_version else ''}"
        f"{task}_{model}_{timestamp}"
    )

    # Use provided name or default
    folder_name = experiment_folder_name or default_name

    # Determine full path
    if experiment_base_dir:
        # Convert to absolute path if relative
        base_dir = Path(experiment_base_dir).resolve()
        base_dir.mkdir(parents=True, exist_ok=True)
        experiment_path = base_dir / folder_name
    else:
        experiment_path = Path(folder_name)

    if experiment_path.exists():
        logger.warning(
            f"‼️ The folder {experiment_path} already exists. Data might be overridden."
        )

    experiment_path.mkdir(parents=True, exist_ok=True)
    return experiment_path


def get_run_params(iteration_type: str, iteration_value: int) -> Tuple[str, int, int]:
    """
    Returns appropriate header, batch_size, and num_concurrency based on iteration_type
    and iteration_value.
    """
    if iteration_type == "batch_size":
        return "Batch Size", iteration_value, 1
    return "Concurrency", 1, iteration_value


def adjust_concurrency_for_target_rate(
    environment: Environment,
    target_rate: float,
    adjustment_interval: float = 5.0,
    min_concurrency: int = 1,
    max_concurrency: int = 1000,
    stop_event: Optional[gevent.event.Event] = None,
) -> gevent.Greenlet:
    """
    Dynamically adjust concurrency to maintain target request rate.
    
    Uses Little's Law: concurrency = request_rate × average_latency
    
    Args:
        environment: Locust Environment instance
        target_rate: Target requests per second
        adjustment_interval: Seconds between concurrency adjustments
        min_concurrency: Minimum allowed concurrency
        max_concurrency: Maximum allowed concurrency
        stop_event: Event to signal when to stop adjusting
        
    Returns:
        Greenlet running the adjustment loop
    """
    if not environment.runner:
        logger.warning("No runner available for concurrency adjustment")
        return None
        
    def adjustment_loop():
        """Background loop that adjusts concurrency based on metrics."""
        last_check_time = time.monotonic()
        last_request_count = 0
        
        while True:
            # Check if we should stop
            if stop_event and stop_event.is_set():
                break
                
            # Wait for adjustment interval
            gevent.sleep(adjustment_interval)
            
            if not environment.runner or not environment.runner.stats:
                continue
                
            current_time = time.monotonic()
            stats = environment.runner.stats
            
            # Get current request count and calculate actual rate
            current_request_count = stats.total.num_requests
            time_delta = current_time - last_check_time
            
            if time_delta < 0.1:  # Too soon, skip this check
                continue
                
            actual_rate = (current_request_count - last_request_count) / time_delta
            last_check_time = current_time
            last_request_count = current_request_count
            
            # Get average response time (E2E latency) from stats
            avg_response_time = stats.total.avg_response_time
            if avg_response_time is None or avg_response_time <= 0:
                # Not enough data yet, skip adjustment
                logger.debug(
                    "Insufficient metrics data for concurrency adjustment, "
                    f"skipping. Requests: {current_request_count}"
                )
                continue
            
            # Calculate required concurrency using Little's Law
            # concurrency = target_rate × average_latency
            required_concurrency = int(target_rate * avg_response_time / 1000.0)  # Convert ms to s
            
            # Apply bounds
            required_concurrency = max(min_concurrency, min(max_concurrency, required_concurrency))
            
            current_users = environment.runner.user_count
            
            # Only adjust if difference is significant (>10% or >2 users)
            if abs(required_concurrency - current_users) > max(2, current_users * 0.1):
                logger.info(
                    f"Adjusting concurrency: {current_users} -> {required_concurrency} "
                    f"(target_rate={target_rate:.2f} req/s, "
                    f"actual_rate={actual_rate:.2f} req/s, "
                    f"avg_latency={avg_response_time:.1f}ms)"
                )
                
                # Adjust concurrency using Locust's runner API
                if required_concurrency > current_users:
                    # Need more users - spawn additional users
                    users_to_spawn = required_concurrency - current_users
                    spawn_rate = min(
                        users_to_spawn,
                        max(1, int(current_users * 0.2) + 1)  # Spawn at most 20% more at once
                    )
                    # Update user count using Locust's internal API
                    try:
                        # Use runner's spawn_users method if available
                        if hasattr(environment.runner, 'spawn_users'):
                            environment.runner.spawn_users(users_to_spawn, spawn_rate=spawn_rate)
                        else:
                            # Fallback: restart with new concurrency
                            # This is disruptive but necessary for Locust compatibility
                            logger.debug(
                                "Restarting runner with new concurrency "
                                f"{current_users} -> {required_concurrency}"
                            )
                            environment.runner.stop()
                            gevent.sleep(0.5)  # Brief pause before restart
                            environment.runner.start(
                                required_concurrency,
                                spawn_rate=spawn_rate
                            )
                    except Exception as e:
                        logger.warning(
                            f"Failed to adjust concurrency: {e}. "
                            "Continuing with current concurrency."
                        )
                else:
                    # Need fewer users - Locust doesn't directly support killing users
                    # Restarting would be too disruptive, so we log and let it adjust
                    # gradually on the next cycle or via natural user completion
                    logger.debug(
                        f"Would reduce concurrency to {required_concurrency}, "
                        "but reduction requires restart (will adjust on next cycle)"
                    )
    
    # Start the adjustment loop in a greenlet
    adjustment_greenlet = gevent.spawn(adjustment_loop)
    return adjustment_greenlet
