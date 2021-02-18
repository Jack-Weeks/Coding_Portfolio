import json
from clustering import cluster

with open('data/cities.json') as json_file:
    city_data = json.load(json_file)
with open('data/libraries.json') as json_file:
    lib_data = json.load(json_file)

city_output = {}
city_list = []
lib_data_list = []
#Extracting City name and Population
for i in range(len(city_data)):
    city_name = city_data[i]['name']
    city_pop = city_data[i]['population']
    city_pop_list = [city_name,city_pop,0]
    city_list.append(city_pop_list)

#Extracting City name and number of books in the library
for i in range(len(lib_data)):
    city_name = lib_data[i]['city']
    book_quant = lib_data[i]['books']
    books_list = [city_name,book_quant]
    lib_data_list.append(books_list)
#Iterating through each city and checking if the city exists in the library list
#if it does, then it adds them together, this accounts for more than one library in each city
for i in range(len(city_list)):
    for j in range(len(lib_data_list)):
        if city_list[i][0] == lib_data_list[j][0]:
            city_list[i][2] += lib_data_list[j][1]

#Constructing Dict
for i in range(len(city_list)):
    output_list = [city_list[i][0], city_list[i][1], city_list[i][2]]
    city_output[output_list[0]] = dict(zip((['population', 'books']), output_list[1:]))

#saving

with open('data/combined_dicts.json', 'w') as output_file:
    json.dump(city_output, output_file, indent= 4)


# Unpacking the Json for cluster files


with open('data/combined_dicts.json') as json_file:
    clustering_data = json.load(json_file)
    key_list = list(clustering_data)
    clustering_data_points = []
    for i in key_list:
        sub_key_list = list(clustering_data[i])
        city_clustering_data = (clustering_data[i][sub_key_list[0]],
                                clustering_data[i][sub_key_list[1]])
        clustering_data_points.append(city_clustering_data)

cluster(clustering_data_points,10,3)



