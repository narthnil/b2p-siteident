# unzip
unzip data.zip -d /tmp/data-b2p

# admin boundaries
for fp in /tmp/data-b2p/data/admin_boundaries/*.zip; do 
    unzip ${fp} -d ./data/admin_boundaries; 
done;
rm -rf ./data/admin_boundaries/__MACOSX/

# country masks
for fp in /tmp/data-b2p/data/country_masks/*.zip; do
    unzip ${fp} -d ./data/country_masks/;
done;
rm -rf  ./data/country_masks/__MACOSX/

# ground truth
cp /tmp/data-b2p/data/ground_truth/* ./data/ground_truth/

# OSM
for fp in /tmp/data-b2p/data/osm/imgs/*.zip; do
    unzip ${fp} -d ./data/osm/imgs/;
done;
for fp in /tmp/data-b2p/data/osm/waterways/*.zip; do 
    unzip ${fp} -d ./data/osm/waterways/; 
done;
for fp in /tmp/data-b2p/data/osm/roads/*.zip; do 
    unzip ${fp} -d ./data/osm/roads/; 
done;

rm -rf ./data/osm/imgs/__MACOSX/
rm -rf ./data/osm/waterways/__MACOSX/
rm -rf ./data/osm/roads/__MACOSX/

# population
for fp in /tmp/data-b2p/data/population/*.zip; do 
    unzip ${fp} -d ./data/population/;
done;
rm -rf ./data/population/__MACOSX/

for fp in /tmp/data-b2p/data/slope_elevation/*.zip; do 
    unzip ${fp} -d ./data/slope_elevation/;
done;
rm -rf  ./data/slope_elevation/__MACOSX/

# bridge type and span data
cp /tmp/data-b2p/data/bridge_type_span_data/* ./data/bridge_type_span_data/
rm -rf ./data/bridge_type_span_data/__MACOSX/


rm -rf /tmp/data-b2p;

# rm -f data.zip