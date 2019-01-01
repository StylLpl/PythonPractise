import pymongo

myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["mydatabase"]
mycol = mydb["LinearRegression"]
# mycol = mydb.create_collection("LinearRegression")



# print(mydb.list_collection_names())

# mydata = {
#     "_id": 1, "x": 2, "y": 4
# }

# x = mycol.insert_one(mydata)

print(mycol.__getitem__("LinearRegression"))
