"""
Robustness and Edge Case Tests for Cyber Monitoring System.
"""

import pytest
from backend.services.scan_service import ScanService
from ml.feature_extractor import extract_url_features

@pytest.fixture
def service():
    return ScanService.get_instance()

def test_edge_case_raw_ipv4(service):
    res = service.scan_url("http://192.168.1.1/admin/login")
    assert res["extractedFeatures"]["has_ip_address"] == 1
    assert res["riskScore"] >= 0

def test_edge_case_raw_ipv6(service):
    res = service.scan_url("http://[2001:db8::1]:8080/index.html")
    assert res["extractedFeatures"]["has_port"] == 1
    assert res["riskScore"] >= 0

def test_edge_case_extremely_long_url(service):
    long_path = "a" * 1500
    long_url = f"https://example.com/test/{long_path}?token={'1234'*50}"
    res = service.scan_url(long_url)
    assert res["extractedFeatures"]["url_length"] > 1500
    assert res["riskLevel"] in ["LOW", "MEDIUM", "HIGH"]

def test_edge_case_heavy_parameter_stuffing(service):
    params = "&".join([f"param_{i}=value_{i}" for i in range(40)])
    url = f"https://example.com/search?{params}"
    res = service.scan_url(url)
    assert res["extractedFeatures"]["num_params"] == 40
    assert res["riskScore"] >= 0

def test_edge_case_punycode_idn(service):
    # Homograph attack simulation (e.g. Cyrillic 'a' -> xn--...)
    puny_url = "http://xn--pple-43d.com/login"
    res = service.scan_url(puny_url)
    assert res["extractedFeatures"]["is_punycode"] == 1

def test_edge_case_embedded_at_symbol(service):
    url = "http://google.com@evil-site.top/verify"
    res = service.scan_url(url)
    assert res["extractedFeatures"]["num_at"] >= 1
    assert res["riskScore"] >= 0

def test_edge_case_no_scheme(service):
    url = "reddit.com/r/cybersecurity"
    res = service.scan_url(url)
    assert res["normalized_url"].startswith("http://")
    assert res["riskLevel"] in ["LOW", "MEDIUM", "HIGH"]
