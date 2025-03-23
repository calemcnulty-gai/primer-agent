#!/bin/bash
set -e

# Configuration
DAILY_API_KEY="pk_c6cd9578-72d5-40fa-8e51-c9109c868f60"
ROOM_NAME="primer"

# Create Daily room
echo "Creating persistent Daily room '${ROOM_NAME}'..."
curl -X POST "https://api.daily.co/v1/rooms" \
     -H "Authorization: Bearer ${DAILY_API_KEY}" \
     -H "Content-Type: application/json" \
     -d "{\"name\": \"${ROOM_NAME}\", \"privacy\": \"public\"}"
echo

echo "Room creation complete."
echo "The persistent Daily room '${ROOM_NAME}' should now be available." 