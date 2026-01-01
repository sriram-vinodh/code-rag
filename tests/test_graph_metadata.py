"""
Tests for graph metadata extraction.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.pipeline.graph_metadata import extract_graph_metadata


def test_structural_row():
    rows = [
        {
            "className": "UserService",
            "filePath": "src/UserService.java",
            "methods": [
                {"name": "login", "signature": "login(String, String)"},
                {"name": "logout", "signature": "logout()"},
            ],
        }
    ]
    meta = extract_graph_metadata(rows).to_dict()
    assert len(meta["classes"]) == 1
    assert len(meta["methods"]) == 2
    names = sorted(m["name"] for m in meta["methods"])
    assert names == ["login", "logout"]


def test_implementation_row_with_code():
    rows = [
        {
            "name": "authenticate",
            "signature": "authenticate(String token)",
            "code": "public boolean authenticate(String token) { ... }",
            "className": "AuthService",
            "filePath": "src/AuthService.java",
        }
    ]
    meta = extract_graph_metadata(rows).to_dict()
    assert meta["methods"][0]["code"].startswith("public boolean authenticate")
    assert meta["methods"][0]["class_name"] == "AuthService"
    assert meta["methods"][0]["file_path"] == "src/AuthService.java"


def test_trace_rows():
    rows = [
        {
            "name": "processPayment",
            "signature": "processPayment(Order order)",
            "className": "PaymentService",
            "filePath": "src/PaymentService.java",
            "calledBy": [
                {"method": "checkout", "className": "CheckoutController", "filePath": "src/CheckoutController.java"}
            ],
            "calls": [
                {"method": "chargeCard", "className": "GatewayClient", "filePath": "src/GatewayClient.java"}
            ],
        }
    ]
    meta = extract_graph_metadata(rows).to_dict()
    method = meta["methods"][0]
    assert method["name"] == "processPayment"
    assert len(method["called_by"]) == 1
    assert len(method["calls"]) == 1


def main():
    test_structural_row()
    test_implementation_row_with_code()
    test_trace_rows()
    print("OK graph_metadata tests passed")
    return True


if __name__ == "__main__":
    success = main()
    raise SystemExit(0 if success else 1)
