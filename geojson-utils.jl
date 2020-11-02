"""
    locations_as_geojson(locations)

Return a list of location as a feature collection of GEOJSON points.

"""
function locations_as_geojson(locations)
    geojson = Dict(
        "type" => "FeatureCollection",
        "features" => map(data -> Dict(
            "type" => "Feature",
            "properties" => Dict(
                "name" => data[1],
                "marker-color" => "#ff0000",
            ),
            "geometry" => Dict(
                "type" => "Point",
                "coordinates" => [data[3], data[2]]
            )
        ), locations)
    );
    JSON.print(geojson)
end


"""
    location_as_h3_polygons(locations)

Return a list of locations converted to a GEOJSON feature collection
consisting of polygons outlining those H3 indexes.

"""
function locations_as_h3_polygons(locations)

    function handle_location(data)
        points = H3.API.h3ToGeoBoundary(location_to_h3_index(data))

        lats = map(x -> rad_to_deg(x.lat), points)
        lons = map(x -> rad_to_deg(x.lon), points)

        coords = collect(zip(lons, lats))

        return Dict(
            "type" => "Feature",
            "properties" => Dict(
                "name" => data[1],
                "fill" => "#ff0000",
            ),
            "geometry" => Dict(
                "type" => "Polygon",
                "coordinates" => [[coords..., coords[1]]]
            )
        )
    end

    geojson = Dict(
        "type" => "FeatureCollection",
        "features" => map(handle_location, locations)
    );
    JSON.print(geojson)
end

