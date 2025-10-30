from locust.env import Environment

import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Tuple

import gevent
import numpy as np

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
    aggregated_metrics_collector: Optional[Any] = None,
    adjustment_interval: float = 5.0,
    min_concurrency: int = 1,
    max_concurrency: int = 1000,
    stop_event: Optional[gevent.event.Event] = None,
) -> gevent.Greenlet:
    """
    Dynamically adjust concurrency to maintain target request rate.
    
    Uses Little's Law: concurrency = request_rate × P90_latency
    (Uses P90 latency instead of average to better handle tail latency)
    
    Args:
        environment: Locust Environment instance
        target_rate: Target requests per second
        aggregated_metrics_collector: Optional metrics collector to access
            P90 latency from collected metrics
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
        check_count = 0
        
        logger.info(
            f"🚀 Starting dynamic concurrency adjustment loop "
            f"(target_rate={target_rate:.2f} req/s, interval={adjustment_interval}s)"
        )
        
        while True:
            # Check if we should stop
            if stop_event and stop_event.is_set():
                logger.info("🛑 Stopping dynamic concurrency adjustment loop")
                break
                
            # Wait for adjustment interval
            gevent.sleep(adjustment_interval)
            check_count += 1
            
            if not environment.runner or not environment.runner.stats:
                logger.debug("No runner or stats available, skipping check")
                continue
                
            current_time = time.monotonic()
            stats = environment.runner.stats
            
            # Get current request count and calculate actual rate
            current_request_count = stats.total.num_requests
            time_delta = current_time - last_check_time
            
            if time_delta < 0.1:  # Too soon, skip this check
                continue
                
            # Calculate actual requests per second
            requests_in_interval = current_request_count - last_request_count
            actual_rate = requests_in_interval / time_delta
            
            # Get average E2E latency from metrics collector if available,
            # otherwise fall back to average from Locust stats
            avg_e2e_latency_s = None
            avg_response_time_ms = stats.total.avg_response_time
            avg_response_time_s = avg_response_time_ms / 1000.0 if avg_response_time_ms else None
            
            # Try to get average latency from aggregated_metrics_collector
            if aggregated_metrics_collector and aggregated_metrics_collector.all_request_metrics:
                # Calculate average from e2e_latency values
                e2e_latencies = [
                    m.e2e_latency
                    for m in aggregated_metrics_collector.all_request_metrics
                    if m.e2e_latency is not None and not m.error_code
                ]
                if e2e_latencies:
                    avg_e2e_latency_s = float(np.mean(e2e_latencies))
            
            # Use average E2E if available, otherwise fall back to Locust's average response time
            latency_to_use_s = avg_e2e_latency_s if avg_e2e_latency_s is not None else avg_response_time_s
            
            # Get error count for completeness
            error_count = stats.total.num_failures
            total_requests = stats.total.num_requests
            
            # Calculate error rate
            error_rate = (error_count / total_requests * 100) if total_requests > 0 else 0
            
            # Get current user count
            current_users = environment.runner.user_count
            
            # Log current status with actual rate and concurrency
            latency_display_ms = latency_to_use_s * 1000 if latency_to_use_s else 0
            
            logger.info(
                f"📊 [Check #{check_count}] Concurrency Adjustment Status:\n"
                f"   Target Rate: {target_rate:.2f} req/s\n"
                f"   Actual Rate: {actual_rate:.2f} req/s "
                f"({requests_in_interval} requests in {time_delta:.1f}s)\n"
                f"   Current Concurrency: {current_users} users\n"
                f"   Avg E2E Latency: {latency_display_ms:.1f}ms ({latency_to_use_s:.3f}s)"
            )
            
            last_check_time = current_time
            last_request_count = current_request_count
            
            if latency_to_use_s is None or latency_to_use_s <= 0:
                # Not enough data yet, skip adjustment
                logger.debug(
                    "⏳ Insufficient metrics data for concurrency adjustment, "
                    f"skipping. Requests: {current_request_count}"
                )
                continue
            
            # Calculate required concurrency using Little's Law
            # concurrency = target_rate × average_latency
            # Round up to ensure we have enough concurrency to achieve the target rate
            required_concurrency = math.ceil(target_rate * latency_to_use_s)
            
            # Apply bounds
            required_concurrency = max(min_concurrency, min(max_concurrency, required_concurrency))
            
            # Log calculated requirements
            logger.info(
                f"🧮 [Check #{check_count}] Required Concurrency Calculation:\n"
                f"   Formula: concurrency = target_rate × average latency\n"
                f"   Calculation: {target_rate:.2f} req/s × {latency_to_use_s:.3f}s "
                f"= {required_concurrency:.1f} users\n"
                f"   Current: {current_users} users | Required: {required_concurrency} users "
                f"| Difference: {required_concurrency - current_users:+d} users"
            )
            
            # Adjust if there's a concurrency difference AND we're more than 1 req/s away from target
            rps_difference = abs(actual_rate - target_rate)
            should_adjust = (
                required_concurrency != current_users
                and rps_difference > 1.0
            )
            
            if should_adjust:
                logger.info(
                    f"⚙️  [Check #{check_count}] Adjusting concurrency: "
                    f"{current_users} → {required_concurrency} users "
                    f"(target: {target_rate:.2f} req/s, actual: {actual_rate:.2f} req/s, "
                    f"difference: {rps_difference:.2f} req/s)"
                )
                
                # Adjust concurrency using Locust's runner API
                try:
                    users_to_spawn = required_concurrency - current_users
                    
                    # Calculate spawn rate for smooth increase
                    spawn_rate_to_use = min(
                        abs(users_to_spawn),
                        max(1, int(current_users * 0.2) + 1)
                    ) if users_to_spawn > 0 else 1
                    
                    # Try setting target_user_count first (newer Locust versions)
                    if hasattr(environment.runner, 'target_user_count'):
                        try:
                            environment.runner.target_user_count = required_concurrency
                            if hasattr(environment.runner, 'spawn_rate'):
                                environment.runner.spawn_rate = spawn_rate_to_use
                            # Wait a bit and check if it worked
                            gevent.sleep(2.0)
                            new_user_count = environment.runner.user_count
                            if new_user_count > current_users:
                                logger.info(
                                    f"   ✅ Users changing: {current_users} → "
                                    f"{new_user_count} users (target: {required_concurrency})"
                                )
                            else:
                                raise ValueError("target_user_count didn't spawn users")
                        except (AttributeError, ValueError, TypeError):
                            # Fall through to stop/start method
                            pass
                    
                    # Fallback: Use stop/start with new concurrency
                    if environment.runner.user_count == current_users:
                        # Check if runner is currently running
                        runner_was_running = False
                        if hasattr(environment.runner, 'state'):
                            runner_was_running = environment.runner.state in ['spawning', 'running']
                        elif hasattr(environment.runner, 'greenlet'):
                            runner_was_running = environment.runner.greenlet is not None
                        
                        if runner_was_running:
                            environment.runner.stop()
                            gevent.sleep(0.5)
                        
                        environment.runner.start(
                            required_concurrency,
                            spawn_rate=spawn_rate_to_use
                        )
                        
                        # Check if it worked after a brief wait
                        gevent.sleep(1.0)
                        new_user_count = environment.runner.user_count
                        if new_user_count > current_users:
                            logger.info(
                                f"   ✅ Users changing: {current_users} → "
                                f"{new_user_count} users (target: {required_concurrency})"
                            )
                    
                except Exception as e:
                    logger.error(
                        f"   ❌ Failed to adjust concurrency: {e}. "
                        f"Exception type: {type(e).__name__}. "
                        "Continuing with current concurrency."
                    )
                    import traceback
                    logger.debug(f"   Traceback: {traceback.format_exc()}")
    
    # Start the adjustment loop in a greenlet
    adjustment_greenlet = gevent.spawn(adjustment_loop)
    return adjustment_greenlet
