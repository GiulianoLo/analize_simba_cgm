"""
Quenching time finder for galaxy star-formation histories.

Implements robust detection of quenching moments in sSFR histories:
- Finds all points where sSFR drops below 1/t (star-formation transition)
- Then finds all points where sSFR drops below 0.2/t (quenching)
- Requires sSFR to stay below 0.2/t for at least 0.2 times the time since the first drop
- Handles multiple quenching events (up to 2)
- Works with oscillatory SFH

Typical usage:
    from simbanator.analysis.quenching import find_quenching_times
    qt, sft, tsq = find_quenching_times(tarr, ssfr)
"""

import numpy as np


def find_quenching_times(tarr, ssfr, min_duration_frac=0.2, max_events=2):
    """
    Find quenching moments in a galaxy sSFR history.

    Parameters
    ----------
    tarr : np.ndarray
        Array of cosmic times (in yr)
    ssfr : np.ndarray
        Array of sSFR values (in 1/yr)
    min_duration_frac : float
        Minimum fraction of time since first drop to stay below 0.2/t (default 0.2)
    max_events : int
        Maximum number of quenching events to return (default 2)

    Returns
    -------
    quench_times : list
        List of quenching moments (in yr)
    sft_times : list
        List of star-formation transition times (in yr)
    time_since_quench : list
        List of time since quenching (in yr)
    """
    tarr = np.asarray(tarr)
    ssfr = np.asarray(ssfr)
    tau_1 = 1 / tarr
    tau_02 = 0.2 / tarr
    log_ssfr = np.log10(ssfr)
    log_tau_1 = np.log10(tau_1)
    log_tau_02 = np.log10(tau_02)

    events = []
    i = 0
    while i < len(tarr) - 1 and len(events) < max_events:
        # Find where sSFR crosses below 1/t
        while i < len(tarr) - 1 and log_ssfr[i] > log_tau_1[i]:
            i += 1
        if i >= len(tarr) - 1:
            break
        sft = tarr[i]
        # Now find where sSFR crosses below 0.2/t
        while i < len(tarr) - 1 and log_ssfr[i] > log_tau_02[i]:
            i += 1
        if i >= len(tarr) - 1:
            break
        qt = tarr[i]
        # Check duration: must stay below 0.2/t for min_duration_frac * (qt - sft)
        duration = min_duration_frac * (qt - sft)
        stay_below = True
        j = i
        while j < len(tarr) and tarr[j] - qt < duration:
            if log_ssfr[j] > log_tau_02[j]:
                stay_below = False
                break
            j += 1
        if stay_below:
            events.append((qt, sft))
            i = j  # Skip ahead to after this event
        else:
            i += 1  # Try next crossing
    quench_times = [qt for qt, sft in events]
    sft_times = [sft for qt, sft in events]
    time_since_quench = [tarr[-1] - qt for qt in quench_times]
    return quench_times, sft_times, time_since_quench
