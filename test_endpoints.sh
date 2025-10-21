#!/bin/bash
# Interactive endpoint tester for Hail Hero

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║         HAIL HERO - INTERACTIVE ENDPOINT TESTER              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

BASE_URL="http://localhost:5000"

# Function to test an endpoint
test_endpoint() {
    local name=$1
    local url=$2
    local method=${3:-GET}

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Testing: $name"
    echo "Endpoint: $method $url"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if [ "$method" = "GET" ]; then
        response=$(curl -s "$url")
        echo "$response" | python3 -m json.tool 2>/dev/null || echo "$response"
    fi

    echo ""
}

# Test all endpoints
echo "🏥 Health & System Endpoints"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
test_endpoint "Health Check" "$BASE_URL/health"
test_endpoint "System Info" "$BASE_URL/api/info"
test_endpoint "API Health" "$BASE_URL/api/v1/health/"

echo ""
echo "📊 Resource Endpoints"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
test_endpoint "List Leads" "$BASE_URL/api/v1/leads/"
test_endpoint "List Inspections" "$BASE_URL/api/v1/inspections/"
test_endpoint "List Contacts" "$BASE_URL/api/v1/contacts/"
test_endpoint "List Events" "$BASE_URL/api/v1/events/"
test_endpoint "List Photos" "$BASE_URL/api/v1/photos/"

echo ""
echo "📚 Documentation"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Swagger JSON schema available at:"
echo "$BASE_URL/api/v1/swagger.json"
echo ""
echo "Interactive Swagger UI available at:"
echo "$BASE_URL/api/v1/docs"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ All endpoints tested!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
