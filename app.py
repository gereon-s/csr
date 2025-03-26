from flask import Flask, render_template, request, jsonify, session
import pandas as pd
import numpy as np
import folium
from folium.plugins import MarkerCluster, Fullscreen
import asyncio
import aiohttp
import re
import os
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv
from aio_georss_gdacs import GdacsFeed
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import openai

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "environmental-catastrophe-detection")


# Make current year available to all templates
@app.context_processor
def inject_current_year():
    return {"current_year": datetime.now().year}


# Initialize OpenAI API if available
def initialize_openai():
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        openai.api_key = api_key
        return True
    return False


# GDACS Data Functions
async def fetch_gdacs_alerts():
    """Fetch only ongoing GDACS alerts using aio-georss-gdacs library."""
    async with aiohttp.ClientSession() as session:
        # Use a neutral point (0, 0) for coordinates
        feed = GdacsFeed(session, (0, 0))
        status, entries = await feed.update()

        if status == "OK":
            # Only include entries that are currently active
            current_entries = []
            for entry in entries:
                # Check if explicitly marked as current
                is_current = getattr(entry, "is_current", None)
                # Check for "to_date" being in the future or not set
                to_date = getattr(entry, "to_date", None)
                is_ongoing = True

                if to_date:
                    try:
                        end_date = datetime.strptime(to_date, "%Y-%m-%d %H:%M:%S")
                        if end_date < datetime.now():
                            is_ongoing = False
                    except (ValueError, TypeError):
                        # If date parsing fails, assume it's ongoing
                        pass

                # Only include if it's marked as current or has no end date or end date is in future
                if (is_current is None or is_current) and is_ongoing:
                    current_entries.append(entry)

            return current_entries
        else:
            print(f"Error fetching GDACS alerts: {status}")
            return []


def process_gdacs_entries(entries):
    """Process GDACS entries into a pandas DataFrame."""

    # Helper function to extract numeric value from various formats
    def extract_numeric_value(value, default=50.0):
        """Extract numeric value from a string, dict, or return the value if already numeric."""
        if isinstance(value, (int, float)):
            return float(value)
        elif isinstance(value, dict) and "value" in value:
            try:
                return float(value["value"])
            except (ValueError, TypeError):
                return default
        elif isinstance(value, str):
            # Try to extract numeric part from strings like "Magnitude 5.8M, Depth:10km"
            match = re.search(r"(\d+\.?\d*)", value)
            if match:
                return float(match.group(1))
        return default

    data = []

    for entry in entries:
        # Map alert level to severity
        if entry.alert_level == "Red":
            severity = "High"
        elif entry.alert_level == "Orange":
            severity = "High"
        else:  # Green or unknown
            severity = "Low"

        # Map event type to our categories
        event_type_map = {
            "EQ": "Earthquake",
            "TC": "Tropical Cyclone",
            "FL": "Flooding",
            "VO": "Volcano",
            "DR": "Drought",
            "WF": "Forest Fire",
            "TS": "Tsunami",
        }

        event_type = event_type_map.get(
            entry.event_type_short, entry.event_type or "Other"
        )

        # Extract coordinates
        lat, lon = None, None
        if entry.coordinates:
            lat, lon = entry.coordinates[0], entry.coordinates[1]

        # Extract affected area - handle different formats
        affected_area = 10.0  # Default value
        if hasattr(entry, "affected_area_km2"):
            affected_area = extract_numeric_value(entry.affected_area_km2, 10.0)

        # Extract risk score or severity as a numeric value
        risk_score = extract_numeric_value(getattr(entry, "severity", 50.0))

        # Create a row for our DataFrame
        data.append(
            {
                "id": entry.external_id,
                "event_type": event_type,
                "latitude": lat,
                "longitude": lon,
                "severity": severity,
                "confidence": extract_numeric_value(
                    getattr(entry, "confidence", 0.9), 0.9
                ),
                "detected_at": entry.from_date
                or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "affected_area_km2": affected_area,
                "status": "Active",  # All events are active since we filtered for current ones
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "detection_method": "Satellite Imagery",
                "risk_score": risk_score,
                "description": entry.description or "",
                "country": entry.country or "Unknown",
                "alert_level": entry.alert_level or "Green",
            }
        )

    # Create DataFrame
    if data:
        df = pd.DataFrame(data)
        return df
    else:
        # Return empty DataFrame with expected columns
        return pd.DataFrame(
            columns=[
                "id",
                "event_type",
                "latitude",
                "longitude",
                "severity",
                "confidence",
                "detected_at",
                "affected_area_km2",
                "status",
                "last_updated",
                "detection_method",
                "risk_score",
                "description",
                "country",
                "alert_level",
            ]
        )


def create_folium_map(df):
    """Create a Folium map with the alerts."""
    # Create a map centered on a neutral location if no data
    if df.empty or not any(pd.notna(df["latitude"]) & pd.notna(df["longitude"])):
        m = folium.Map(location=[0, 0], zoom_start=2, control_scale=True)
        return m

    # Filter for entries with valid coordinates
    valid_df = df[pd.notna(df["latitude"]) & pd.notna(df["longitude"])]

    if valid_df.empty:
        m = folium.Map(location=[0, 0], zoom_start=2, control_scale=True)
        return m

    # Create a map centered on data
    m = folium.Map(
        location=[valid_df["latitude"].mean(), valid_df["longitude"].mean()],
        zoom_start=2,
        control_scale=True,
    )

    # Create feature groups for each event type
    event_types = valid_df["event_type"].unique()
    feature_groups = {
        event_type: folium.FeatureGroup(name=event_type) for event_type in event_types
    }

    # Add markers to appropriate feature groups
    for _, row in valid_df.iterrows():
        # Determine marker color based on alert level
        if row["alert_level"] == "Red":
            color = "red"
        elif row["alert_level"] == "Orange":
            color = "orange"
        else:  # Green or unknown
            color = "green"

        # Determine icon based on event type
        icon_map = {
            "Earthquake": "bolt",
            "Tropical Cyclone": "cloud",
            "Flooding": "tint",
            "Volcano": "fire",
            "Drought": "sun",
            "Forest Fire": "fire",
            "Tsunami": "water",
        }
        icon = icon_map.get(row["event_type"], "info-sign")

        # Create popup content
        popup_html = f"""
        <div style="font-family: Arial; max-width: 300px;">
            <h4>{row['event_type']} - {row['country']}</h4>
            <strong>Alert Level:</strong> {row['alert_level']}<br>
            <strong>Status:</strong> {row['status']}<br>
            <strong>Detected:</strong> {row['detected_at']}<br>
            <strong>Description:</strong> {row['description'][:150]}...
        </div>
        """

        # Add marker to map
        folium.Marker(
            location=[row["latitude"], row["longitude"]],
            popup=folium.Popup(popup_html, max_width=300),
            icon=folium.Icon(color=color, icon=icon, prefix="fa"),
            tooltip=f"{row['event_type']} ({row['alert_level']})",
        ).add_to(feature_groups[row["event_type"]])

    # Add feature groups to map
    for group in feature_groups.values():
        group.add_to(m)

    # Add layer control
    folium.LayerControl().add_to(m)

    return m


# Geocoding function to get coordinates from location name
def get_coordinates(location_name):
    """Get coordinates from a location name."""
    try:
        # Process input for countries first
        region_coordinates = {
            "Europe": (48.8566, 19.3522),
            "North America": (39.8283, -98.5795),
            "South America": (-23.5505, -58.4371),
            "Asia": (34.0479, 100.6197),
            "Africa": (8.7832, 25.5085),
            "Australia": (-25.2744, 133.7751),
            "Oceania": (-8.7832, 143.4317),
            "USA": (37.0902, -95.7129),
            "US": (37.0902, -95.7129),
            "United States": (37.0902, -95.7129),
            "UK": (55.3781, -3.4360),
            "United Kingdom": (55.3781, -3.4360),
            "Brazil": (-14.2350, -51.9253),
            "India": (20.5937, 78.9629),
            "China": (35.8617, 104.1954),
            "Russia": (61.5240, 105.3188),
            "Canada": (56.1304, -106.3468),
            "Australia": (-25.2744, 133.7751),
            "Japan": (36.2048, 138.2529),
            "Germany": (51.1657, 10.4515),
            "France": (46.6034, 1.8883),
            "Italy": (41.8719, 12.5674),
            "Spain": (40.4637, -3.7492),
            "Mexico": (23.6345, -102.5528),
            "South Africa": (-30.5595, 22.9375),
            "Nigeria": (9.0820, 8.6753),
            "Egypt": (26.8206, 30.8025),
            "Kenya": (-0.0236, 37.9062),
            "Ghana": (7.9465, -1.0232),
            "Morocco": (31.7917, -7.0926),
            "Tanzania": (-6.3690, 34.8888),
            "Uganda": (1.3733, 32.2903),
            "Zambia": (-13.1339, 27.8493),
            "Zimbabwe": (-19.0154, 29.1549),
            "Argentina": (-38.4161, -63.6167),
            "Chile": (-35.6751, -71.5430),
            "Colombia": (4.5709, -74.2973),
            "Peru": (-9.1899, -75.0152),
            "Venezuela": (6.4238, -66.5897),
            "Ecuador": (-1.8312, -78.1834),
        }

        # Check if location is a known region
        if location_name in region_coordinates:
            return region_coordinates[location_name]

        # For more specific locations, use geocoding
        geolocator = Nominatim(user_agent="gdacs_app")
        location = geolocator.geocode(location_name, timeout=10)

        if location:
            return location.latitude, location.longitude
        else:
            return None, None
    except Exception as e:
        print(f"Error geocoding location: {str(e)}")
        return None, None


# Find disasters near a location
def find_disasters_near_location(df, lat, lon, radius_km=2000):
    if lat is None or lon is None or df.empty:
        return pd.DataFrame()

    # Create a new DataFrame with distance column
    near_df = df.copy()

    # Calculate distance to the specified location
    distances = []
    for idx, row in near_df.iterrows():
        if pd.notna(row["latitude"]) and pd.notna(row["longitude"]):
            distance = geodesic(
                (lat, lon), (row["latitude"], row["longitude"])
            ).kilometers
            distances.append(distance)
        else:
            distances.append(float("inf"))

    near_df["distance_km"] = distances

    # Filter by radius
    result = near_df[near_df["distance_km"] <= radius_km].sort_values("distance_km")
    return result


# AI Chat function
def get_ai_response(user_query, user_location, nearby_disasters):
    has_openai = initialize_openai()

    if not has_openai:
        return "Sorry, AI features are currently unavailable. Please set your OpenAI API key in the environment variables."

    try:
        # Format disaster information for the AI
        disasters_info = ""
        if not nearby_disasters.empty:
            disasters_info = "Here are the disasters near the user's location:\n"
            for _, disaster in nearby_disasters.iterrows():
                disasters_info += f"- {disaster['event_type']} in {disaster['country']}, {disaster['distance_km']:.0f}km away. Alert level: {disaster['alert_level']}. {disaster['description'][:100]}...\n"
        else:
            disasters_info = "There are no disasters detected near the user's location."

        # Create the prompt for OpenAI
        prompt = f"""You are a disaster information assistant. The user is asking about disasters near {user_location}.

User query: {user_query}

{disasters_info}

Please provide a helpful, conversational response addressing the user's query based on this GDACS disaster data. 
Keep your response brief and focused on the disaster information relevant to their location.
If there are no disasters nearby, reassure them but mention that they should still stay informed about global events.
"""

        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful disaster information assistant based on GDACS data.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=300,
            temperature=0.7,
        )

        return response.choices[0].message.content

    except Exception as e:
        print(f"Error getting AI response: {str(e)}")
        return f"Sorry, I encountered an error while processing your question. Please try again later."


# Flood Prediction Functions
def load_flood_data():
    """Load all flood prediction results from JSON file."""
    try:
        with open("flood_detection_results.json", "r") as f:
            data = json.load(f)  # Load all data

        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Ensure date is in datetime format for filtering
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])

        # Sort by date (most recent first)
        if "date" in df.columns:
            df = df.sort_values("date", ascending=False)

        return df
    except Exception as e:
        print(f"Error loading flood detection data: {str(e)}")
        # Return empty DataFrame with expected columns
        return pd.DataFrame(
            columns=[
                "location_id",
                "location_folder",
                "date",
                "filename",
                "latitude",
                "longitude",
                "true_label",
                "predicted_label",
                "confidence",
                "coordinates",
                "environmental_impact",
                "economic_impact",
                "social_impact",
                "affected_area_km2",
                "total_impact",
                "status",
                "risk_score",
            ]
        )


def create_flood_map(df, use_clustering=True, max_points=None):
    """Create a Folium map with flood prediction markers.

    Args:
        df: DataFrame with flood prediction data
        use_clustering: Whether to use marker clustering
        max_points: Maximum number of points to display (None for all)
    """
    # Create a map centered on a neutral location if no data
    if df.empty or not any(pd.notna(df["latitude"]) & pd.notna(df["longitude"])):
        m = folium.Map(location=[0, 0], zoom_start=2, control_scale=True)
        return m

    # Filter for entries with valid coordinates
    valid_df = df[pd.notna(df["latitude"]) & pd.notna(df["longitude"])].copy()

    if valid_df.empty:
        m = folium.Map(location=[0, 0], zoom_start=2, control_scale=True)
        return m

    # If max_points is specified, perform intelligent sampling to maintain geographical diversity
    if max_points and len(valid_df) > max_points:
        # Group by approximate location (rounded coordinates)
        valid_df["lat_bin"] = np.round(valid_df["latitude"], 1)
        valid_df["lon_bin"] = np.round(valid_df["longitude"], 1)
        grouped = valid_df.groupby(["lat_bin", "lon_bin"])

        # Take a proportional number of samples from each group
        sampled_df = pd.DataFrame()
        for _, group in grouped:
            # Calculate how many samples to take from this group
            # Formula: (group_size / total_size) * max_points, at least 1 point per group
            sample_size = max(1, int((len(group) / len(valid_df)) * max_points))
            # Sample from the group, take all if sample_size >= group size
            if len(group) <= sample_size:
                sampled_df = pd.concat([sampled_df, group])
            else:
                sampled_df = pd.concat([sampled_df, group.sample(sample_size)])

        # If we still have too many points, take a random sample up to max_points
        if len(sampled_df) > max_points:
            sampled_df = sampled_df.sample(max_points)

        valid_df = sampled_df

    # Create a map centered on data
    m = folium.Map(
        location=[valid_df["latitude"].mean(), valid_df["longitude"].mean()],
        zoom_start=3,
        control_scale=True,
    )

    # Create feature groups for active floods and non-floods
    if use_clustering:
        # Import MarkerCluster if needed
        from folium.plugins import MarkerCluster

        active_floods = MarkerCluster(name="Active Floods", overlay=True, control=True)
        no_floods = MarkerCluster(name="No Flood Detected", overlay=True, control=True)
    else:
        active_floods = folium.FeatureGroup(name="Active Floods")
        no_floods = folium.FeatureGroup(name="No Flood Detected")

    # Add markers to appropriate feature groups
    for _, row in valid_df.iterrows():
        # Determine marker color and icon based on prediction
        if row["predicted_label"] == 1:  # Active flood
            color = "blue"
            icon = "tint"
            feature_group = active_floods
        else:  # No flood
            color = "green"
            icon = "check"
            feature_group = no_floods

        # Create popup content
        popup_html = f"""
        <div style="font-family: Arial; max-width: 300px;">
            <h4>Flood Prediction - {row.get('location_id', 'Unknown')}</h4>
            <strong>Status:</strong> {row.get('status', 'Unknown')}<br>
            <strong>Date:</strong> {row.get('date').strftime('%Y-%m-%d') if isinstance(row.get('date'), pd.Timestamp) else row.get('date', 'Unknown')}<br>
            <strong>Confidence:</strong> {row.get('confidence', 0)*100:.1f}%<br>
            <strong>Risk Score:</strong> {row.get('risk_score', 0):.1f}<br>
            <strong>Affected Area:</strong> {row.get('affected_area_km2', 0):.2f} km²<br>
            <strong>Total Impact:</strong> {row.get('total_impact', 0):.2f}/10<br>
            <strong>Coordinates:</strong> {row["latitude"]:.4f}, {row["longitude"]:.4f}
        </div>
        """

        # Add marker
        folium.Marker(
            location=[row["latitude"], row["longitude"]],
            popup=folium.Popup(popup_html, max_width=300),
            icon=folium.Icon(color=color, icon=icon, prefix="fa"),
            tooltip=f"Flood Prediction ({row.get('confidence', 0)*100:.0f}% confidence)",
        ).add_to(feature_group)

    # Add feature groups to map
    active_floods.add_to(m)
    no_floods.add_to(m)

    # Add layer control
    folium.LayerControl().add_to(m)

    # Add fullscreen option
    from folium.plugins import Fullscreen

    Fullscreen().add_to(m)

    return m


# Routes
@app.route("/")
def index():
    # Fetch GDACS data
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    entries = loop.run_until_complete(fetch_gdacs_alerts())
    loop.close()

    # Process the data
    df = process_gdacs_entries(entries)

    # Calculate summary statistics
    active_count = len(df) if not df.empty else 0
    high_severity_count = len(df[df["severity"] == "High"]) if not df.empty else 0
    countries_count = len(df["country"].unique()) if not df.empty else 0
    event_types_count = len(df["event_type"].unique()) if not df.empty else 0

    # Create the map
    disaster_map = create_folium_map(df)
    map_html = disaster_map._repr_html_()

    # Get high severity events
    high_severity = (
        df[df["severity"] == "High"].sort_values("detected_at", ascending=False).head(3)
    )

    return render_template(
        "index.html",
        active_count=active_count,
        high_severity_count=high_severity_count,
        countries_count=countries_count,
        event_types_count=event_types_count,
        map_html=map_html,
        high_severity=high_severity.to_dict("records"),
        last_updated=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )


@app.route("/flood-predictions")
def flood_predictions():
    # Load flood data
    df = load_flood_data()

    # Get page parameter from request
    page = request.args.get("page", 1, type=int)
    items_per_page = request.args.get("per_page", 20, type=int)
    show_only_floods = request.args.get("show_only_floods", "true") == "true"
    use_clustering = request.args.get("use_clustering", "true") == "true"
    max_points = request.args.get("max_points", type=int)  # None by default

    if not df.empty and "date" in df.columns:
        # Calculate summary statistics (using all data)
        active_floods = len(df[df["predicted_label"] == 1])
        avg_confidence = (
            df[df["predicted_label"] == 1]["confidence"].mean()
            if active_floods > 0
            else 0
        )
        total_area = (
            df[df["predicted_label"] == 1]["affected_area_km2"].sum()
            if active_floods > 0
            else 0
        )
        avg_impact = (
            df[df["predicted_label"] == 1]["total_impact"].mean()
            if active_floods > 0
            else 0
        )

        # Filter for floods if requested
        if show_only_floods:
            filtered_df = df[df["predicted_label"] == 1]
        else:
            filtered_df = df

        # Calculate total pages
        total_items = len(filtered_df)
        total_pages = (total_items + items_per_page - 1) // items_per_page

        # Adjust page if out of bounds
        if page < 1:
            page = 1
        elif page > total_pages and total_pages > 0:
            page = total_pages

        # Get paginated data
        start_idx = (page - 1) * items_per_page
        end_idx = min(start_idx + items_per_page, total_items)

        # Get current page data
        paginated_data = filtered_df.iloc[start_idx:end_idx]

        # Get map data - use ALL data with valid coordinates
        map_data = filtered_df[
            pd.notna(filtered_df["latitude"]) & pd.notna(filtered_df["longitude"])
        ]

        # Create the map with clustering for better performance
        flood_map = create_flood_map(
            map_data, use_clustering=use_clustering, max_points=max_points
        )
        map_html = flood_map._repr_html_()

        # Get high impact floods
        high_impact_floods = (
            df[(df["predicted_label"] == 1) & (df["total_impact"] >= 7.0)]
            .sort_values("total_impact", ascending=False)
            .head(5)
        )

        # Add debug info
        valid_coord_count = len(map_data)
        map_points_count = (
            len(map_data)
            if max_points is None or len(map_data) <= max_points
            else max_points
        )

        # Determine the display message
        if use_clustering:
            map_display_message = (
                f"Using marker clustering for {valid_coord_count} locations"
            )
        elif max_points is not None and valid_coord_count > max_points:
            map_display_message = f"Displaying {map_points_count} geographically diverse locations (sampled from {valid_coord_count} total)"
        else:
            map_display_message = (
                f"Displaying all {map_points_count} locations with valid coordinates"
            )

        return render_template(
            "flood_predictions.html",
            active_floods=active_floods,
            avg_confidence=avg_confidence,
            total_area=total_area,
            avg_impact=avg_impact,
            map_html=map_html,
            high_impact_floods=high_impact_floods.to_dict("records"),
            flood_data=paginated_data.to_dict("records"),
            total_pages=total_pages,
            current_page=page,
            show_only_floods=show_only_floods,
            total_items=total_items,
            valid_coord_count=valid_coord_count,
            map_points_count=map_points_count,
            map_display_message=map_display_message,
            use_clustering=use_clustering,
            max_points=max_points,
            last_updated=(
                df["date"].max().strftime("%Y-%m-%d")
                if not df.empty and isinstance(df["date"].max(), pd.Timestamp)
                else "N/A"
            ),
        )
    else:
        return render_template(
            "flood_predictions.html",
            active_floods=0,
            avg_confidence=0,
            total_area=0,
            avg_impact=0,
            map_html="<p>No flood data available</p>",
            high_impact_floods=[],
            flood_data=[],
            total_pages=0,
            current_page=1,
            show_only_floods=True,
            total_items=0,
            valid_coord_count=0,
            map_points_count=0,
            map_display_message="No map data available",
            use_clustering=True,
            max_points=None,
            last_updated="N/A",
        )


@app.route("/api/location", methods=["POST"])
def process_location():
    data = request.json
    location = data.get("location", "")

    # Get coordinates
    lat, lon = get_coordinates(location)

    if lat is None or lon is None:
        return jsonify({"success": False, "message": "Location not found"})

    # Fetch GDACS data
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    entries = loop.run_until_complete(fetch_gdacs_alerts())
    loop.close()

    # Process the data
    df = process_gdacs_entries(entries)

    # Find nearby disasters
    nearby = find_disasters_near_location(df, lat, lon)

    # Store in session
    session["user_location"] = location
    session["user_coordinates"] = (lat, lon)

    return jsonify(
        {
            "success": True,
            "location": location,
            "coordinates": [lat, lon],
            "nearby_count": len(nearby),
            "nearby_disasters": nearby.to_dict("records"),
        }
    )


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json
    query = data.get("query", "")

    if "user_location" not in session or "user_coordinates" not in session:
        return jsonify({"success": False, "message": "Please set your location first"})

    location = session["user_location"]
    lat, lon = session["user_coordinates"]

    # Fetch GDACS data
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    entries = loop.run_until_complete(fetch_gdacs_alerts())
    loop.close()

    # Process the data
    df = process_gdacs_entries(entries)

    # Find nearby disasters
    nearby = find_disasters_near_location(df, lat, lon)

    # Get AI response
    response = get_ai_response(query, location, nearby)

    return jsonify({"success": True, "query": query, "response": response})


@app.route("/api/flood-data/<format>")
def download_flood_data(format):
    """Endpoint to download flood data in CSV or JSON format"""
    # Load flood data
    df = load_flood_data()

    if format == "csv":
        response = app.response_class(
            df.to_csv(index=False),
            mimetype="text/csv",
            headers={
                "Content-Disposition": "attachment;filename=flood_predictions_export.csv"
            },
        )
        return response
    elif format == "json":
        response = app.response_class(
            df.to_json(orient="records", date_format="iso"),
            mimetype="application/json",
            headers={
                "Content-Disposition": "attachment;filename=flood_predictions_export.json"
            },
        )
        return response
    else:
        return jsonify({"error": "Invalid format requested"})


if __name__ == "__main__":
    app.run(debug=True)
