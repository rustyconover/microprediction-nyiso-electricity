# Implement a bounding box that is based on the borders of a H3
# hexagon for an array of locations.
struct Bounds
    lat_min
    lat_max
    lon_min
    lon_max

    function Bounds(points::Array{Tuple{String,Float64,Float64}})
        points = reduce(vcat, (map(l -> H3.API.h3ToGeoBoundary(location_to_h3_index(l)), points)))

        lats = map(x -> rad_to_deg(x.lat), points)
        lons = map(x -> rad_to_deg(x.lon), points)

        new(minimum(lats),
            maximum(lats),
            minimum(lons),
            maximum(lons)
        )
    end
end

intersects(self::Bounds, lat, lon) = lat >= self.lat_min &&
                                     lat <= self.lat_max &&
                                     lon >= self.lon_min &&
                                     lon <= self.lon_max


