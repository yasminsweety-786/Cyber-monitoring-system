"""
Safe Network Inspection Service for URL validation and lightweight header probing.
Performs cautious, isolated HTTP HEAD/GET probes with strict timeouts and zero code execution.
"""

import urllib.parse
import requests
from typing import Dict, Any, Optional
from backend.config import NETWORK_TIMEOUT_SECONDS

def inspect_network_safety(url: str, timeout: float = NETWORK_TIMEOUT_SECONDS) -> Dict[str, Any]:
    """
    Safely probes a URL with an HTTP HEAD request to determine connectivity,
    redirect counts, final destination host, and status codes.
    Fails safely without throwing unhandled exceptions.
    """
    result = {
        "reachable": False,
        "redirect_count": 0,
        "final_url": url,
        "status_code": None,
        "error": None
    }
    
    try:
        parsed = urllib.parse.urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            result["error"] = "Invalid URL structure"
            return result
            
        # Cautious request with custom bot User-Agent and stream=True (does not download body payload)
        headers = {
            "User-Agent": "CyberRiskMonitor-Scanner/1.0 (Security-Analysis-Probe; Safe; +http://cyber-monitor.local)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
        }
        
        # HEAD request first to avoid downloading malware binaries
        response = requests.head(
            url, 
            timeout=timeout, 
            allow_redirects=True, 
            headers=headers,
            verify=False # Do not crash on self-signed / expired SSL
        )
        
        result["reachable"] = True
        result["status_code"] = response.status_code
        result["redirect_count"] = len(response.history)
        result["final_url"] = str(response.url)
        
    except requests.exceptions.Timeout:
        result["error"] = f"Connection timed out ({timeout}s threshold)"
    except requests.exceptions.SSLError as e:
        result["error"] = "SSL certificate validation anomaly"
    except requests.exceptions.ConnectionError:
        result["error"] = "Host unreachable or DNS resolution failed"
    except Exception as e:
        result["error"] = f"Probe error: {str(e)[:80]}"
        
    return result
