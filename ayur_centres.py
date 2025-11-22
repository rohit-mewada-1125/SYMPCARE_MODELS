import requests
from geopy.distance import geodesic

def fetch_hospitals_from_osm(lat, lon, radius=5000):
    # 1. Get nearby hospitals using Overpass API
    url = "https://overpass-api.de/api/interpreter"
    
    query = f"""
    [out:json];
    (
      node["amenity"="hospital"](around:{radius},{lat},{lon});
      way["amenity"="hospital"](around:{radius},{lat},{lon});
      relation["amenity"="hospital"](around:{radius},{lat},{lon});
    );
    out body;
    """
    
    headers = {
        'User-Agent': 'YourAppName/1.0 (https://yourwebsite.com)'  # Custom User-Agent
    }

    response = requests.get(url, params={'data': query}, headers=headers)
    hospitals = []

    if response.status_code == 200:
        try:
            data = response.json()
            if data.get('elements'):
                destinations = []
                # Collect destinations for distance calculation
                for place in data['elements']:
                    hospital_lat = place.get('lat', 'N/A')
                    hospital_lon = place.get('lon', 'N/A')
                    destinations.append(f"{hospital_lat},{hospital_lon}")

                # 2. Get distances from the user's location via Distance Matrix API
                dist_url = "https://maps.googleapis.com/maps/api/distancematrix/json"
                dist_params = {
                    "origins": f"{lat},{lon}",
                    "destinations": "|".join(destinations),
                    "key": "YOUR_GOOGLE_API_KEY"  # Replace with your Google API key
                }

                dist_response = requests.get(dist_url, params=dist_params)
                dist_data = dist_response.json()

                # Ensure we have valid distance data
                if "rows" in dist_data and dist_data["rows"]:
                    distances = dist_data["rows"][0].get("elements", [])
                else:
                    distances = []

                # 3. Build the final list of hospitals with additional details
                for i, place in enumerate(data['elements']):
                    name = place.get('tags', {}).get('name', 'Unnamed')
                    address = place.get('tags', {}).get('addr:full', 'Address not available')
                    phone = place.get('tags', {}).get('contact:phone', 'Phone not available')
                    hospital_lat = place.get('lat', 'N/A')
                    hospital_lon = place.get('lon', 'N/A')

                    # Calculate distance from the user's location to the hospital
                    hospital_location = (hospital_lat, hospital_lon)
                    user_location = (lat, lon)
                    distance = geodesic(user_location, hospital_location).kilometers

                    # Add the hospital details to the list
                    hospitals.append({
                        "name": name,
                        "address": address,
                        "lat": hospital_lat,
                        "lon": hospital_lon,
                        "distance_text": distances[i]["distance"]["text"] if i < len(distances) and distances[i].get("distance") else f"{distance:.2f} kilometers",
                        "duration_text": distances[i]["duration"]["text"] if i < len(distances) and distances[i].get("duration") else "N/A",
                        "phone": phone
                    })
            else:
                print("No hospitals found nearby.")
        except ValueError as e:
            print(f"Error parsing JSON: {e}")
    else:
        print(f"Error: {response.status_code}, {response.text}")

    # Sort hospitals by distance and get the top 10
    hospitals_sorted = sorted(hospitals, key=lambda x: geodesic((lat, lon), (x["lat"], x["lon"])).kilometers)[:10]

    return hospitals_sorted


# lat_bhopal, lon_bhopal = 23.2599, 77.4126
# hospitals = fetch_hospitals_from_osm(lat_bhopal, lon_bhopal)

# # Print out hospital details for the top 10 nearest hospitals
# for hospital in hospitals:
#     print(f"Name: {hospital['name']}")
#     print(f"Address: {hospital['address']}")
#     print(f"Latitude: {hospital['lat']}")
#     print(f"Longitude: {hospital['lon']}")
#     print(f"Distance: {hospital['distance_text']}")
#     print(f"Duration: {hospital['duration_text']}")
#     print(f"Phone: {hospital['phone']}")
#     print("-----------------------------")
