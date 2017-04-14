var documenterSearchIndex = {"docs": [

{
    "location": "index.html#",
    "page": "Home",
    "title": "Home",
    "category": "page",
    "text": ""
},

{
    "location": "index.html#Impute-1",
    "page": "Home",
    "title": "Impute",
    "category": "section",
    "text": "(Image: stable) (Image: latest) (Image: Build Status) (Image: Build status) (Image: codecov)Impute.jl provides various data imputation methods for Arrays, NullableArrays and DataArrays(for vectors and matrices), as well as DataFrames and DataTables."
},

{
    "location": "index.html#Installation-1",
    "page": "Home",
    "title": "Installation",
    "category": "section",
    "text": "Pkg.clone(\"https://github.com/invenia/Impute.jl\")"
},

{
    "location": "index.html#Features-1",
    "page": "Home",
    "title": "Features",
    "category": "section",
    "text": "Vectors and Matrices\nNullableArrays and DataArrays\nChaining of methods"
},

{
    "location": "index.html#Methods-1",
    "page": "Home",
    "title": "Methods",
    "category": "section",
    "text": "drop - remove missing\nlocf - last observation carried forward\nnocb - next observation carried backward\ninterp - linear interpolation of values in vector\nfill - replace with a specific value or a function which returns a value given the existing vector with missing values dropped."
},

{
    "location": "index.html#Quickstart-1",
    "page": "Home",
    "title": "Quickstart",
    "category": "section",
    "text": "We'll start by imputing NaN values in 1-dimension vector.julia> using Impute\n\njulia> a = collect(1.0:1.0:20.0)\n20-element Array{Float64,1}:\n  1.0\n  2.0\n  3.0\n  4.0\n  5.0\n  6.0\n  7.0\n  8.0\n  9.0\n 10.0\n 11.0\n 12.0\n 13.0\n 14.0\n 15.0\n 16.0\n 17.0\n 18.0\n 19.0\n 20.0\n\njulia> a[[2, 3, 7]] = NaN\nNaNThe most common approach to missing data is to remove them.julia> impute(a, :drop; limit=0.2)\n17-element Array{Float64,1}:\n  1.0\n  4.0\n  5.0\n  6.0\n  8.0\n  9.0\n 10.0\n 11.0\n 12.0\n 13.0\n 14.0\n 15.0\n 16.0\n 17.0\n 18.0\n 19.0\n 20.0But we may want use linear interpolation, filling, etcjulia> impute(a, :interp; limit=0.2)\n20-element Array{Float64,1}:\n  1.0\n  2.0\n  3.0\n  4.0\n  5.0\n  6.0\n  7.0\n  8.0\n  9.0\n 10.0\n 11.0\n 12.0\n 13.0\n 14.0\n 15.0\n 16.0\n 17.0\n 18.0\n 19.0\n 20.0\n\njulia> impute(a, :fill; limit=0.2)\n20-element Array{Float64,1}:\n  1.0\n 11.6471\n 11.6471\n  4.0\n  5.0\n  6.0\n 11.6471\n  8.0\n  9.0\n 10.0\n 11.0\n 12.0\n 13.0\n 14.0\n 15.0\n 16.0\n 17.0\n 18.0\n 19.0\n 20.0\n\njulia> impute(a, :locf; limit=0.2)\n20-element Array{Float64,1}:\n  1.0\n  1.0\n  1.0\n  4.0\n  5.0\n  6.0\n  6.0\n  8.0\n  9.0\n 10.0\n 11.0\n 12.0\n 13.0\n 14.0\n 15.0\n 16.0\n 17.0\n 18.0\n 19.0\n 20.0\n\njulia> impute(a, :nocb; limit=0.2)\n20-element Array{Float64,1}:\n  1.0\n  4.0\n  4.0\n  4.0\n  5.0\n  6.0\n  8.0\n  8.0\n  9.0\n 10.0\n 11.0\n 12.0\n 13.0\n 14.0\n 15.0\n 16.0\n 17.0\n 18.0\n 19.0\n 20.0These operations also work on NullableArrays.julia> using NullableArrays\n\njulia> b = NullableArray(a)\n20-element NullableArrays.NullableArray{Float64,1}:\n 1.0\n NaN\n NaN\n 4.0\n 5.0\n 6.0\n NaN\n 8.0\n 9.0\n 10.0\n 11.0\n 12.0\n 13.0\n 14.0\n 15.0\n 16.0\n 17.0\n 18.0\n 19.0\n 20.0\n\njulia> b[[2, 3, 7]] = Nullable()\nNullable{Union{}}()\n\njulia> impute(a, :interp; limit=0.2)\n20-element Array{Float64,1}:\n  1.0\n  2.0\n  3.0\n  4.0\n  5.0\n  6.0\n  7.0\n  8.0\n  9.0\n 10.0\n 11.0\n 12.0\n 13.0\n 14.0\n 15.0\n 16.0\n 17.0\n 18.0\n 19.0\n 20.0We can also perform these operations on DataFrames and DataTables.julia> using DataFrames\n\njulia> using RDatasets\n\njulia> df = dataset(\"boot\", \"neuro\")\n2814\nSymbol[:V1,:V2,:V3,:V4,:V5,:V6]\n6\n469×6 DataFrames.DataFrame\n│ Row │ V1     │ V2     │ V3     │ V4    │ V5    │ V6    │\n├─────┼────────┼────────┼────────┼───────┼───────┼───────┤\n│ 1   │ NA     │ -203.7 │ -84.1  │ 18.5  │ NA    │ NA    │\n│ 2   │ NA     │ -203.0 │ -97.8  │ 25.8  │ 134.7 │ NA    │\n│ 3   │ NA     │ -249.0 │ -92.1  │ 27.8  │ 177.1 │ NA    │\n│ 4   │ NA     │ -231.5 │ -97.5  │ 27.0  │ 150.3 │ NA    │\n│ 5   │ NA     │ NA     │ -130.1 │ 25.8  │ 160.0 │ NA    │\n│ 6   │ NA     │ -223.1 │ -70.7  │ 62.1  │ 197.5 │ NA    │\n│ 7   │ NA     │ -164.8 │ -12.2  │ 76.8  │ 202.8 │ NA    │\n│ 8   │ NA     │ -221.6 │ -81.9  │ 27.5  │ 144.5 │ NA    │\n│ 9   │ NA     │ -153.7 │ -17.0  │ 76.1  │ 222.4 │ NA    │\n│ 10  │ NA     │ -184.7 │ -47.3  │ 74.4  │ 208.9 │ NA    │\n│ 11  │ NA     │ NA     │ -148.8 │ 11.4  │ 137.7 │ NA    │\n│ 12  │ NA     │ -197.6 │ -6.4   │ 137.1 │ NA    │ NA    │\n│ 13  │ NA     │ -247.8 │ -35.4  │ 80.9  │ 229.5 │ NA    │\n│ 14  │ NA     │ -227.0 │ -104.7 │ 20.2  │ 140.2 │ NA    │\n│ 15  │ -233.6 │ -115.9 │ -10.5  │ 70.0  │ 202.6 │ NA    │\n│ 16  │ NA     │ -232.4 │ -100.6 │ 16.8  │ 145.1 │ NA    │\n│ 17  │ NA     │ -199.4 │ -58.2  │ 29.1  │ 184.4 │ NA    │\n│ 18  │ NA     │ -195.7 │ -89.5  │ 26.4  │ 142.7 │ NA    │\n│ 19  │ NA     │ -180.1 │ -65.0  │ 27.3  │ 171.1 │ NA    │\n│ 20  │ NA     │ NA     │ -85.2  │ 27.1  │ NA    │ NA    │\n│ 21  │ NA     │ -217.3 │ -77.1  │ 27.6  │ 151.5 │ NA    │\n│ 22  │ NA     │ -139.7 │ -15.8  │ 83.0  │ 215.5 │ NA    │\n│ 23  │ -249.6 │ -132.8 │ -14.1  │ 78.1  │ 205.7 │ NA    │\n│ 24  │ NA     │ -152.7 │ -36.9  │ 29.7  │ 149.8 │ NA    │\n│ 25  │ NA     │ -224.1 │ -81.9  │ 29.1  │ 172.2 │ NA    │\n│ 26  │ NA     │ NA     │ -235.8 │ 6.0   │ 144.4 │ NA    │\n│ 27  │ NA     │ -202.8 │ -45.1  │ 84.0  │ 227.3 │ NA    │\n│ 28  │ -240.9 │ -138.4 │ -21.5  │ 73.4  │ 210.6 │ NA    │\n│ 29  │ -247.1 │ -128.2 │ -31.3  │ 29.2  │ 143.1 │ NA    │\n│ 30  │ NA     │ -185.4 │ -80.3  │ 23.9  │ 115.8 │ 222.7 │\n│ 31  │ NA     │ -182.5 │ -75.8  │ 27.5  │ 165.2 │ NA    │\n│ 32  │ NA     │ -202.2 │ -99.1  │ 23.8  │ 136.3 │ 242.5 │\n│ 33  │ NA     │ -193.3 │ -82.6  │ 26.3  │ 160.5 │ NA    │\n│ 34  │ NA     │ -189.4 │ -63.3  │ 27.6  │ 136.8 │ NA    │\n│ 35  │ NA     │ -149.0 │ -31.0  │ 73.5  │ 187.8 │ NA    │\n│ 36  │ NA     │ -162.4 │ -26.5  │ 72.6  │ NA    │ NA    │\n│ 37  │ NA     │ -213.4 │ -107.2 │ 24.7  │ 158.5 │ NA    │\n⋮\n│ 432 │ NA     │ -156.2 │ -32.9  │ 63.3  │ 182.8 │ NA    │\n│ 433 │ NA     │ -220.6 │ -114.2 │ 9.7   │ 106.4 │ 227.9 │\n│ 434 │ -219.9 │ -120.9 │ -1.3   │ 99.5  │ 207.6 │ NA    │\n│ 435 │ NA     │ -240.5 │ -110.3 │ 26.1  │ 142.8 │ NA    │\n│ 436 │ NA     │ -239.6 │ -121.4 │ 2.9   │ 124.9 │ NA    │\n│ 437 │ NA     │ -139.8 │ -7.3   │ 121.0 │ NA    │ NA    │\n│ 438 │ NA     │ -212.0 │ -66.2  │ 50.4  │ 178.2 │ NA    │\n│ 439 │ NA     │ -232.7 │ -109.2 │ 18.4  │ 127.5 │ NA    │\n│ 440 │ NA     │ -236.3 │ -115.1 │ 5.1   │ 109.0 │ 212.0 │\n│ 441 │ -241.2 │ -107.1 │ -9.1   │ 95.1  │ 198.6 │ NA    │\n│ 442 │ -226.7 │ -143.8 │ -30.4  │ 75.8  │ 196.6 │ NA    │\n│ 443 │ NA     │ -131.8 │ -26.5  │ 64.7  │ 177.2 │ NA    │\n│ 444 │ NA     │ -144.9 │ -0.9   │ 105.3 │ 230.9 │ NA    │\n│ 445 │ NA     │ -214.0 │ -81.8  │ 66.1  │ 191.3 │ NA    │\n│ 446 │ NA     │ -210.6 │ -94.3  │ 16.7  │ 125.5 │ 239.7 │\n│ 447 │ -215.8 │ -114.8 │ -18.4  │ 65.3  │ 171.6 │ 249.7 │\n│ 448 │ NA     │ -156.0 │ -14.0  │ 113.7 │ 249.3 │ NA    │\n│ 449 │ NA     │ -210.5 │ -41.9  │ NA    │ NA    │ NA    │\n│ 450 │ NA     │ -189.2 │ -72.0  │ 56.8  │ 133.8 │ 246.7 │\n│ 451 │ NA     │ -214.2 │ -102.2 │ 5.5   │ 75.6  │ 154.3 │\n│ 452 │ -219.6 │ -107.9 │ -16.0  │ 101.7 │ 186.0 │ NA    │\n│ 453 │ NA     │ -153.0 │ -38.0  │ 61.3  │ 144.4 │ 245.9 │\n│ 454 │ NA     │ -179.8 │ -63.4  │ 56.0  │ 157.5 │ NA    │\n│ 455 │ NA     │ -174.5 │ -44.8  │ 73.3  │ 179.7 │ NA    │\n│ 456 │ NA     │ -206.8 │ -108.9 │ 3.7   │ 102.1 │ 210.3 │\n│ 457 │ NA     │ -169.5 │ -79.7  │ 27.9  │ 129.4 │ 242.8 │\n│ 458 │ -222.2 │ -104.6 │ -2.4   │ 84.3  │ 204.7 │ NA    │\n│ 459 │ -236.3 │ -124.0 │ -6.8   │ 95.7  │ 196.0 │ NA    │\n│ 460 │ NA     │ -216.5 │ -90.2  │ 27.8  │ 138.9 │ NA    │\n│ 461 │ NA     │ -163.2 │ -43.6  │ 69.5  │ 173.9 │ NA    │\n│ 462 │ NA     │ -207.3 │ -88.3  │ 9.6   │ 104.1 │ 218.0 │\n│ 463 │ -242.6 │ -142.0 │ -21.8  │ 69.8  │ 148.7 │ NA    │\n│ 464 │ -235.9 │ -128.8 │ -33.1  │ 68.8  │ 177.1 │ NA    │\n│ 465 │ NA     │ -140.8 │ -38.7  │ 58.1  │ 186.3 │ NA    │\n│ 466 │ NA     │ -149.5 │ -40.3  │ 62.8  │ 139.7 │ 242.5 │\n│ 467 │ -247.6 │ -157.8 │ -53.3  │ 28.3  │ 122.9 │ 227.6 │\n│ 468 │ NA     │ -154.9 │ -50.8  │ 28.1  │ 119.9 │ 201.1 │\n│ 469 │ NA     │ -180.7 │ -70.9  │ 33.7  │ 114.8 │ 222.5 │\n\njulia> drop(df)\n4×6 DataFrames.DataFrame\n│ Row │ V1     │ V2     │ V3    │ V4   │ V5    │ V6    │\n├─────┼────────┼────────┼───────┼──────┼───────┼───────┤\n│ 1   │ -247.0 │ -132.2 │ -18.8 │ 28.2 │ 81.4  │ 237.9 │\n│ 2   │ -234.0 │ -140.8 │ -56.5 │ 28.0 │ 114.3 │ 222.9 │\n│ 3   │ -215.8 │ -114.8 │ -18.4 │ 65.3 │ 171.6 │ 249.7 │\n│ 4   │ -247.6 │ -157.8 │ -53.3 │ 28.3 │ 122.9 │ 227.6 │\n\njulia> interp(df)\n469×6 DataFrames.DataFrame\n│ Row │ V1       │ V2      │ V3     │ V4    │ V5     │ V6      │\n├─────┼──────────┼─────────┼────────┼───────┼────────┼─────────┤\n│ 1   │ NA       │ -203.7  │ -84.1  │ 18.5  │ NA     │ NA      │\n│ 2   │ NA       │ -203.0  │ -97.8  │ 25.8  │ 134.7  │ NA      │\n│ 3   │ NA       │ -249.0  │ -92.1  │ 27.8  │ 177.1  │ NA      │\n│ 4   │ NA       │ -231.5  │ -97.5  │ 27.0  │ 150.3  │ NA      │\n│ 5   │ NA       │ -227.3  │ -130.1 │ 25.8  │ 160.0  │ NA      │\n│ 6   │ NA       │ -223.1  │ -70.7  │ 62.1  │ 197.5  │ NA      │\n│ 7   │ NA       │ -164.8  │ -12.2  │ 76.8  │ 202.8  │ NA      │\n│ 8   │ NA       │ -221.6  │ -81.9  │ 27.5  │ 144.5  │ NA      │\n│ 9   │ NA       │ -153.7  │ -17.0  │ 76.1  │ 222.4  │ NA      │\n│ 10  │ NA       │ -184.7  │ -47.3  │ 74.4  │ 208.9  │ NA      │\n│ 11  │ NA       │ -191.15 │ -148.8 │ 11.4  │ 137.7  │ NA      │\n│ 12  │ NA       │ -197.6  │ -6.4   │ 137.1 │ 183.6  │ NA      │\n│ 13  │ NA       │ -247.8  │ -35.4  │ 80.9  │ 229.5  │ NA      │\n│ 14  │ NA       │ -227.0  │ -104.7 │ 20.2  │ 140.2  │ NA      │\n│ 15  │ -233.6   │ -115.9  │ -10.5  │ 70.0  │ 202.6  │ NA      │\n│ 16  │ -235.6   │ -232.4  │ -100.6 │ 16.8  │ 145.1  │ NA      │\n│ 17  │ -237.6   │ -199.4  │ -58.2  │ 29.1  │ 184.4  │ NA      │\n│ 18  │ -239.6   │ -195.7  │ -89.5  │ 26.4  │ 142.7  │ NA      │\n│ 19  │ -241.6   │ -180.1  │ -65.0  │ 27.3  │ 171.1  │ NA      │\n│ 20  │ -243.6   │ -198.7  │ -85.2  │ 27.1  │ 161.3  │ NA      │\n│ 21  │ -245.6   │ -217.3  │ -77.1  │ 27.6  │ 151.5  │ NA      │\n│ 22  │ -247.6   │ -139.7  │ -15.8  │ 83.0  │ 215.5  │ NA      │\n│ 23  │ -249.6   │ -132.8  │ -14.1  │ 78.1  │ 205.7  │ NA      │\n│ 24  │ -247.86  │ -152.7  │ -36.9  │ 29.7  │ 149.8  │ NA      │\n│ 25  │ -246.12  │ -224.1  │ -81.9  │ 29.1  │ 172.2  │ NA      │\n│ 26  │ -244.38  │ -213.45 │ -235.8 │ 6.0   │ 144.4  │ NA      │\n│ 27  │ -242.64  │ -202.8  │ -45.1  │ 84.0  │ 227.3  │ NA      │\n│ 28  │ -240.9   │ -138.4  │ -21.5  │ 73.4  │ 210.6  │ NA      │\n│ 29  │ -247.1   │ -128.2  │ -31.3  │ 29.2  │ 143.1  │ NA      │\n│ 30  │ -247.093 │ -185.4  │ -80.3  │ 23.9  │ 115.8  │ 222.7   │\n│ 31  │ -247.086 │ -182.5  │ -75.8  │ 27.5  │ 165.2  │ 232.6   │\n│ 32  │ -247.079 │ -202.2  │ -99.1  │ 23.8  │ 136.3  │ 242.5   │\n│ 33  │ -247.071 │ -193.3  │ -82.6  │ 26.3  │ 160.5  │ 242.082 │\n│ 34  │ -247.064 │ -189.4  │ -63.3  │ 27.6  │ 136.8  │ 241.664 │\n│ 35  │ -247.057 │ -149.0  │ -31.0  │ 73.5  │ 187.8  │ 241.245 │\n│ 36  │ -247.05  │ -162.4  │ -26.5  │ 72.6  │ 173.15 │ 240.827 │\n│ 37  │ -247.043 │ -213.4  │ -107.2 │ 24.7  │ 158.5  │ 240.409 │\n⋮\n│ 432 │ -219.99  │ -156.2  │ -32.9  │ 63.3  │ 182.8  │ 232.0   │\n│ 433 │ -219.945 │ -220.6  │ -114.2 │ 9.7   │ 106.4  │ 227.9   │\n│ 434 │ -219.9   │ -120.9  │ -1.3   │ 99.5  │ 207.6  │ 225.629 │\n│ 435 │ -222.943 │ -240.5  │ -110.3 │ 26.1  │ 142.8  │ 223.357 │\n│ 436 │ -225.986 │ -239.6  │ -121.4 │ 2.9   │ 124.9  │ 221.086 │\n│ 437 │ -229.029 │ -139.8  │ -7.3   │ 121.0 │ 151.55 │ 218.814 │\n│ 438 │ -232.071 │ -212.0  │ -66.2  │ 50.4  │ 178.2  │ 216.543 │\n│ 439 │ -235.114 │ -232.7  │ -109.2 │ 18.4  │ 127.5  │ 214.271 │\n│ 440 │ -238.157 │ -236.3  │ -115.1 │ 5.1   │ 109.0  │ 212.0   │\n│ 441 │ -241.2   │ -107.1  │ -9.1   │ 95.1  │ 198.6  │ 216.617 │\n│ 442 │ -226.7   │ -143.8  │ -30.4  │ 75.8  │ 196.6  │ 221.233 │\n│ 443 │ -224.52  │ -131.8  │ -26.5  │ 64.7  │ 177.2  │ 225.85  │\n│ 444 │ -222.34  │ -144.9  │ -0.9   │ 105.3 │ 230.9  │ 230.467 │\n│ 445 │ -220.16  │ -214.0  │ -81.8  │ 66.1  │ 191.3  │ 235.083 │\n│ 446 │ -217.98  │ -210.6  │ -94.3  │ 16.7  │ 125.5  │ 239.7   │\n│ 447 │ -215.8   │ -114.8  │ -18.4  │ 65.3  │ 171.6  │ 249.7   │\n│ 448 │ -216.56  │ -156.0  │ -14.0  │ 113.7 │ 249.3  │ 248.7   │\n│ 449 │ -217.32  │ -210.5  │ -41.9  │ 85.25 │ 191.55 │ 247.7   │\n│ 450 │ -218.08  │ -189.2  │ -72.0  │ 56.8  │ 133.8  │ 246.7   │\n│ 451 │ -218.84  │ -214.2  │ -102.2 │ 5.5   │ 75.6   │ 154.3   │\n│ 452 │ -219.6   │ -107.9  │ -16.0  │ 101.7 │ 186.0  │ 200.1   │\n│ 453 │ -220.033 │ -153.0  │ -38.0  │ 61.3  │ 144.4  │ 245.9   │\n│ 454 │ -220.467 │ -179.8  │ -63.4  │ 56.0  │ 157.5  │ 234.033 │\n│ 455 │ -220.9   │ -174.5  │ -44.8  │ 73.3  │ 179.7  │ 222.167 │\n│ 456 │ -221.333 │ -206.8  │ -108.9 │ 3.7   │ 102.1  │ 210.3   │\n│ 457 │ -221.767 │ -169.5  │ -79.7  │ 27.9  │ 129.4  │ 242.8   │\n│ 458 │ -222.2   │ -104.6  │ -2.4   │ 84.3  │ 204.7  │ 237.84  │\n│ 459 │ -236.3   │ -124.0  │ -6.8   │ 95.7  │ 196.0  │ 232.88  │\n│ 460 │ -237.875 │ -216.5  │ -90.2  │ 27.8  │ 138.9  │ 227.92  │\n│ 461 │ -239.45  │ -163.2  │ -43.6  │ 69.5  │ 173.9  │ 222.96  │\n│ 462 │ -241.025 │ -207.3  │ -88.3  │ 9.6   │ 104.1  │ 218.0   │\n│ 463 │ -242.6   │ -142.0  │ -21.8  │ 69.8  │ 148.7  │ 224.125 │\n│ 464 │ -235.9   │ -128.8  │ -33.1  │ 68.8  │ 177.1  │ 230.25  │\n│ 465 │ -239.8   │ -140.8  │ -38.7  │ 58.1  │ 186.3  │ 236.375 │\n│ 466 │ -243.7   │ -149.5  │ -40.3  │ 62.8  │ 139.7  │ 242.5   │\n│ 467 │ -247.6   │ -157.8  │ -53.3  │ 28.3  │ 122.9  │ 227.6   │\n│ 468 │ NA       │ -154.9  │ -50.8  │ 28.1  │ 119.9  │ 201.1   │\n│ 469 │ NA       │ -180.7  │ -70.9  │ 33.7  │ 114.8  │ 222.5   │Finally, we can also chain imputation methods together. As we saw in the last example linear interpolation can interpolate missing values at the head or tail of the array (or column).julia> chain(df, Impute.Interpolate(), Impute.LOCF(), Impute.NOCB(); limit=1.0)\n469×6 DataFrames.DataFrame\n│ Row │ V1       │ V2      │ V3     │ V4    │ V5     │ V6      │\n├─────┼──────────┼─────────┼────────┼───────┼────────┼─────────┤\n│ 1   │ -233.6   │ -203.7  │ -84.1  │ 18.5  │ 134.7  │ 222.7   │\n│ 2   │ -233.6   │ -203.0  │ -97.8  │ 25.8  │ 134.7  │ 222.7   │\n│ 3   │ -233.6   │ -249.0  │ -92.1  │ 27.8  │ 177.1  │ 222.7   │\n│ 4   │ -233.6   │ -231.5  │ -97.5  │ 27.0  │ 150.3  │ 222.7   │\n│ 5   │ -233.6   │ -227.3  │ -130.1 │ 25.8  │ 160.0  │ 222.7   │\n│ 6   │ -233.6   │ -223.1  │ -70.7  │ 62.1  │ 197.5  │ 222.7   │\n│ 7   │ -233.6   │ -164.8  │ -12.2  │ 76.8  │ 202.8  │ 222.7   │\n│ 8   │ -233.6   │ -221.6  │ -81.9  │ 27.5  │ 144.5  │ 222.7   │\n│ 9   │ -233.6   │ -153.7  │ -17.0  │ 76.1  │ 222.4  │ 222.7   │\n│ 10  │ -233.6   │ -184.7  │ -47.3  │ 74.4  │ 208.9  │ 222.7   │\n│ 11  │ -233.6   │ -191.15 │ -148.8 │ 11.4  │ 137.7  │ 222.7   │\n│ 12  │ -233.6   │ -197.6  │ -6.4   │ 137.1 │ 183.6  │ 222.7   │\n│ 13  │ -233.6   │ -247.8  │ -35.4  │ 80.9  │ 229.5  │ 222.7   │\n│ 14  │ -233.6   │ -227.0  │ -104.7 │ 20.2  │ 140.2  │ 222.7   │\n│ 15  │ -233.6   │ -115.9  │ -10.5  │ 70.0  │ 202.6  │ 222.7   │\n│ 16  │ -235.6   │ -232.4  │ -100.6 │ 16.8  │ 145.1  │ 222.7   │\n│ 17  │ -237.6   │ -199.4  │ -58.2  │ 29.1  │ 184.4  │ 222.7   │\n│ 18  │ -239.6   │ -195.7  │ -89.5  │ 26.4  │ 142.7  │ 222.7   │\n│ 19  │ -241.6   │ -180.1  │ -65.0  │ 27.3  │ 171.1  │ 222.7   │\n│ 20  │ -243.6   │ -198.7  │ -85.2  │ 27.1  │ 161.3  │ 222.7   │\n│ 21  │ -245.6   │ -217.3  │ -77.1  │ 27.6  │ 151.5  │ 222.7   │\n│ 22  │ -247.6   │ -139.7  │ -15.8  │ 83.0  │ 215.5  │ 222.7   │\n│ 23  │ -249.6   │ -132.8  │ -14.1  │ 78.1  │ 205.7  │ 222.7   │\n│ 24  │ -247.86  │ -152.7  │ -36.9  │ 29.7  │ 149.8  │ 222.7   │\n│ 25  │ -246.12  │ -224.1  │ -81.9  │ 29.1  │ 172.2  │ 222.7   │\n│ 26  │ -244.38  │ -213.45 │ -235.8 │ 6.0   │ 144.4  │ 222.7   │\n│ 27  │ -242.64  │ -202.8  │ -45.1  │ 84.0  │ 227.3  │ 222.7   │\n│ 28  │ -240.9   │ -138.4  │ -21.5  │ 73.4  │ 210.6  │ 222.7   │\n│ 29  │ -247.1   │ -128.2  │ -31.3  │ 29.2  │ 143.1  │ 222.7   │\n│ 30  │ -247.093 │ -185.4  │ -80.3  │ 23.9  │ 115.8  │ 222.7   │\n│ 31  │ -247.086 │ -182.5  │ -75.8  │ 27.5  │ 165.2  │ 232.6   │\n│ 32  │ -247.079 │ -202.2  │ -99.1  │ 23.8  │ 136.3  │ 242.5   │\n│ 33  │ -247.071 │ -193.3  │ -82.6  │ 26.3  │ 160.5  │ 242.082 │\n│ 34  │ -247.064 │ -189.4  │ -63.3  │ 27.6  │ 136.8  │ 241.664 │\n│ 35  │ -247.057 │ -149.0  │ -31.0  │ 73.5  │ 187.8  │ 241.245 │\n│ 36  │ -247.05  │ -162.4  │ -26.5  │ 72.6  │ 173.15 │ 240.827 │\n│ 37  │ -247.043 │ -213.4  │ -107.2 │ 24.7  │ 158.5  │ 240.409 │\n⋮\n│ 432 │ -219.99  │ -156.2  │ -32.9  │ 63.3  │ 182.8  │ 232.0   │\n│ 433 │ -219.945 │ -220.6  │ -114.2 │ 9.7   │ 106.4  │ 227.9   │\n│ 434 │ -219.9   │ -120.9  │ -1.3   │ 99.5  │ 207.6  │ 225.629 │\n│ 435 │ -222.943 │ -240.5  │ -110.3 │ 26.1  │ 142.8  │ 223.357 │\n│ 436 │ -225.986 │ -239.6  │ -121.4 │ 2.9   │ 124.9  │ 221.086 │\n│ 437 │ -229.029 │ -139.8  │ -7.3   │ 121.0 │ 151.55 │ 218.814 │\n│ 438 │ -232.071 │ -212.0  │ -66.2  │ 50.4  │ 178.2  │ 216.543 │\n│ 439 │ -235.114 │ -232.7  │ -109.2 │ 18.4  │ 127.5  │ 214.271 │\n│ 440 │ -238.157 │ -236.3  │ -115.1 │ 5.1   │ 109.0  │ 212.0   │\n│ 441 │ -241.2   │ -107.1  │ -9.1   │ 95.1  │ 198.6  │ 216.617 │\n│ 442 │ -226.7   │ -143.8  │ -30.4  │ 75.8  │ 196.6  │ 221.233 │\n│ 443 │ -224.52  │ -131.8  │ -26.5  │ 64.7  │ 177.2  │ 225.85  │\n│ 444 │ -222.34  │ -144.9  │ -0.9   │ 105.3 │ 230.9  │ 230.467 │\n│ 445 │ -220.16  │ -214.0  │ -81.8  │ 66.1  │ 191.3  │ 235.083 │\n│ 446 │ -217.98  │ -210.6  │ -94.3  │ 16.7  │ 125.5  │ 239.7   │\n│ 447 │ -215.8   │ -114.8  │ -18.4  │ 65.3  │ 171.6  │ 249.7   │\n│ 448 │ -216.56  │ -156.0  │ -14.0  │ 113.7 │ 249.3  │ 248.7   │\n│ 449 │ -217.32  │ -210.5  │ -41.9  │ 85.25 │ 191.55 │ 247.7   │\n│ 450 │ -218.08  │ -189.2  │ -72.0  │ 56.8  │ 133.8  │ 246.7   │\n│ 451 │ -218.84  │ -214.2  │ -102.2 │ 5.5   │ 75.6   │ 154.3   │\n│ 452 │ -219.6   │ -107.9  │ -16.0  │ 101.7 │ 186.0  │ 200.1   │\n│ 453 │ -220.033 │ -153.0  │ -38.0  │ 61.3  │ 144.4  │ 245.9   │\n│ 454 │ -220.467 │ -179.8  │ -63.4  │ 56.0  │ 157.5  │ 234.033 │\n│ 455 │ -220.9   │ -174.5  │ -44.8  │ 73.3  │ 179.7  │ 222.167 │\n│ 456 │ -221.333 │ -206.8  │ -108.9 │ 3.7   │ 102.1  │ 210.3   │\n│ 457 │ -221.767 │ -169.5  │ -79.7  │ 27.9  │ 129.4  │ 242.8   │\n│ 458 │ -222.2   │ -104.6  │ -2.4   │ 84.3  │ 204.7  │ 237.84  │\n│ 459 │ -236.3   │ -124.0  │ -6.8   │ 95.7  │ 196.0  │ 232.88  │\n│ 460 │ -237.875 │ -216.5  │ -90.2  │ 27.8  │ 138.9  │ 227.92  │\n│ 461 │ -239.45  │ -163.2  │ -43.6  │ 69.5  │ 173.9  │ 222.96  │\n│ 462 │ -241.025 │ -207.3  │ -88.3  │ 9.6   │ 104.1  │ 218.0   │\n│ 463 │ -242.6   │ -142.0  │ -21.8  │ 69.8  │ 148.7  │ 224.125 │\n│ 464 │ -235.9   │ -128.8  │ -33.1  │ 68.8  │ 177.1  │ 230.25  │\n│ 465 │ -239.8   │ -140.8  │ -38.7  │ 58.1  │ 186.3  │ 236.375 │\n│ 466 │ -243.7   │ -149.5  │ -40.3  │ 62.8  │ 139.7  │ 242.5   │\n│ 467 │ -247.6   │ -157.8  │ -53.3  │ 28.3  │ 122.9  │ 227.6   │\n│ 468 │ -247.6   │ -154.9  │ -50.8  │ 28.1  │ 119.9  │ 201.1   │\n│ 469 │ -247.6   │ -180.7  │ -70.9  │ 33.7  │ 114.8  │ 222.5   │"
},

{
    "location": "api/impute.html#Impute.ImputeError",
    "page": "Impute",
    "title": "Impute.ImputeError",
    "category": "Type",
    "text": "ImputeError{T} <: Exception\n\nIs thrown by impute methods when the limit of imputable values has been exceeded.\n\nFields\n\nmsg::T - the message to print.\n\n\n\n"
},

{
    "location": "api/impute.html#Impute.chain!-Tuple{Union{AbstractArray, DataFrames.DataFrame, DataTables.DataTable},Function,Vararg{Impute.Imputor,N} where N}",
    "page": "Impute",
    "title": "Impute.chain!",
    "category": "Method",
    "text": "chain!(data::Dataset, missing::Function, imputors::Imputor...; kwargs...)\n\nCreates a Chain with imputors and calls impute!(imputor, missing, data; kwargs...)\n\n\n\n"
},

{
    "location": "api/impute.html#Impute.chain!-Tuple{Union{AbstractArray, DataFrames.DataFrame, DataTables.DataTable},Vararg{Impute.Imputor,N} where N}",
    "page": "Impute",
    "title": "Impute.chain!",
    "category": "Method",
    "text": "chain!(data::Dataset, imputors::Imputor...; kwargs...)\n\nCreates a Chain with imputors and calls impute!(imputor, data; kwargs...)\n\n\n\n"
},

{
    "location": "api/impute.html#Impute.chain-Tuple{Union{AbstractArray, DataFrames.DataFrame, DataTables.DataTable},Vararg{Any,N} where N}",
    "page": "Impute",
    "title": "Impute.chain",
    "category": "Method",
    "text": "chain(data::Dataset, args...; kwargs...)\n\nCopies the data before calling chain!(data, args...; kwargs...)\n\n\n\n"
},

{
    "location": "api/impute.html#Impute.drop!-Tuple{Union{AbstractArray, DataFrames.DataFrame, DataTables.DataTable}}",
    "page": "Impute",
    "title": "Impute.drop!",
    "category": "Method",
    "text": "drop!(data::Dataset; limit=1.0)\n\nUtility method for impute!(data, :drop; limit=limit)\n\n\n\n"
},

{
    "location": "api/impute.html#Impute.impute!-Tuple{Union{AbstractArray, DataFrames.DataFrame, DataTables.DataTable},Function,Symbol,Vararg{Any,N} where N}",
    "page": "Impute",
    "title": "Impute.impute!",
    "category": "Method",
    "text": "impute!(data::Dataset, missing::Function, method::Symbol=:interp, args...; limit::Float64=0.1)\n\nCreates the appropriate Imputor type and Context (using missing function) in order to call impute!(imputor::Imputor, ctx::Context, data::Dataset) with them.\n\nArguments\n\ndata::Dataset: the datset containing missing elements we should impute.\nmissing::Function: the missing data function to use\nmethod::Symbol: the imputation method to use   (options: [:drop, :fill, :interp, :locf, :nocb])\nargs::Any...: any arguments you should pass to the Imputor constructor.\nlimit::Float64: missing data ratio limit/threshold (default: 0.1)\n\n\n\n"
},

{
    "location": "api/impute.html#Impute.impute!-Tuple{Union{AbstractArray, DataFrames.DataFrame, DataTables.DataTable},Symbol,Vararg{Any,N} where N}",
    "page": "Impute",
    "title": "Impute.impute!",
    "category": "Method",
    "text": "impute!(data::Dataset, method::Symbol=:interp, args...; limit::Float64=0.1)\n\nLooks up the Imputor type for the method, creates it and calls impute!(imputor::Imputor, data::Dataset, limit::Float64) with it.\n\nArguments\n\ndata::Dataset: the datset containing missing elements we should impute.\nmethod::Symbol: the imputation method to use   (options: [:drop, :fill, :interp, :locf, :nocb])\nargs::Any...: any arguments you should pass to the Imputor constructor.\nlimit::Float64: missing data ratio limit/threshold (default: 0.1)\n\n\n\n"
},

{
    "location": "api/impute.html#Impute.impute-Tuple{Union{AbstractArray, DataFrames.DataFrame, DataTables.DataTable},Vararg{Any,N} where N}",
    "page": "Impute",
    "title": "Impute.impute",
    "category": "Method",
    "text": "impute(data::Dataset, args...; kwargs...)\n\nCopies the data before calling impute!(new_data, args...; kwargs...)\n\n\n\n"
},

{
    "location": "api/impute.html#Impute.interp!-Tuple{Union{AbstractArray, DataFrames.DataFrame, DataTables.DataTable}}",
    "page": "Impute",
    "title": "Impute.interp!",
    "category": "Method",
    "text": "interp!(data::Dataset; limit=1.0)\n\nUtility method for impute!(data, :interp; limit=limit)\n\n\n\n"
},

{
    "location": "api/impute.html#Impute.interp-Tuple{Union{AbstractArray, DataFrames.DataFrame, DataTables.DataTable}}",
    "page": "Impute",
    "title": "Impute.interp",
    "category": "Method",
    "text": "interp(data::Dataset; limit=1.0)\n\nUtility method for impute(data, :interp; limit=limit)\n\n\n\n"
},

{
    "location": "api/impute.html#Base.drop-Tuple{Union{AbstractArray, DataFrames.DataFrame, DataTables.DataTable}}",
    "page": "Impute",
    "title": "Base.drop",
    "category": "Method",
    "text": "drop(data::Dataset; limit=1.0)\n\nUtility method for impute(data, :drop; limit=limit)\n\n\n\n"
},

{
    "location": "api/impute.html#",
    "page": "Impute",
    "title": "Impute",
    "category": "page",
    "text": "Modules = [Impute]\nPrivate = true\nPages = [\"Impute.jl\"]\nOrder = [:module, :constant, :type, :function]"
},

{
    "location": "api/context.html#Impute.Context",
    "page": "Context",
    "title": "Impute.Context",
    "category": "Type",
    "text": "Context\n\nStores common summary information for all Imputor types.\n\nFields\n\nnum::Int: number of observations\ncount::Int: number of missing values found\nlimit::Float64: allowable limit for missing values to impute\nmissing::Function: returns a Bool if the value counts as missing.\n\n\n\n"
},

{
    "location": "api/context.html#Base.findfirst-Union{Tuple{Impute.Context,AbstractArray{T,1}}, Tuple{T}} where T",
    "page": "Context",
    "title": "Base.findfirst",
    "category": "Method",
    "text": "findfirst{T<:Any}(ctx::Context, data::AbstractArray{T, 1}) -> Int\n\nReturns the first not missing index in data.\n\nArguments\n\nctx::Context: the context to pass into is_missing\ndata::AbstractArray{T, 1}: the data array to search\n\nReturns\n\nInt: the first index in data that isn't missing\n\n\n\n"
},

{
    "location": "api/context.html#Base.findlast-Union{Tuple{Impute.Context,AbstractArray{T,1}}, Tuple{T}} where T",
    "page": "Context",
    "title": "Base.findlast",
    "category": "Method",
    "text": "findlast{T<:Any}(ctx::Context, data::AbstractArray{T, 1}) -> Int\n\nReturns the last not missing index in data.\n\nArguments\n\nctx::Context: the context to pass into is_missing\ndata::AbstractArray{T, 1}: the data array to search\n\nReturns\n\nInt: the last index in data that isn't missing\n\n\n\n"
},

{
    "location": "api/context.html#Base.findnext-Union{Tuple{Impute.Context,AbstractArray{T,1},Int64}, Tuple{T}} where T",
    "page": "Context",
    "title": "Base.findnext",
    "category": "Method",
    "text": "findnext{T<:Any}(ctx::Context, data::AbstractArray{T, 1}) -> Int\n\nReturns the next not missing index in data.\n\nArguments\n\nctx::Context: the context to pass into is_missing\ndata::AbstractArray{T, 1}: the data array to search\n\nReturns\n\nInt: the next index in data that isn't missing\n\n\n\n"
},

{
    "location": "api/context.html#Impute.is_missing-Tuple{Impute.Context,Any}",
    "page": "Context",
    "title": "Impute.is_missing",
    "category": "Method",
    "text": "is_missing(ctx::Context, x) -> Bool\n\nUses ctx.missing to determine if x is missing. If x is a data row or an abstract array then is_missing will return true if ctx.missing returns true for any element. The ctx.count is increased whenever whenever we return true and if ctx.count / ctx.num exceeds our ctx.limit we throw an ImputeError\n\nArguments\n\nctx::Context: the contextual information about missing information.\nx: the value to check (may be an single values, abstract array or row)\n\n\n\n"
},

{
    "location": "api/context.html#",
    "page": "Context",
    "title": "Context",
    "category": "page",
    "text": "Modules = [Impute]\nPages = [\"context.jl\"]\nOrder = [:module, :constant, :type, :function]"
},

{
    "location": "api/imputors.html#Impute.impute!",
    "page": "Imputors",
    "title": "Impute.impute!",
    "category": "Function",
    "text": "impute!(imp::Imputor, data::Dataset, limit::Float64=0.1)\n\nCreates a Context using information about data. These include\n\nmissing data function:\n\nisnull: if NullableArray or DataTable\nisna: if DataArray or DataFrame\nisnan: for anything else.\n\nnumber of elements: *(size(data)...)\n\nArguments\n\nimp::Imputor: the Imputor method to use\ndata::Dataset: the data to impute\nlimit::Float64: missing data ratio limit/threshold (default: 0.1)\n\nReturn\n\nDataset: the input data with values imputed.\n\n\n\n"
},

{
    "location": "api/imputors.html#Impute.impute!-Tuple{Impute.Imputor,Impute.Context,Union{DataFrames.DataFrame, DataTables.DataTable}}",
    "page": "Imputors",
    "title": "Impute.impute!",
    "category": "Method",
    "text": "impute!{T<:Any}(imp::Imputor, ctx::Context, data::Table)\n\nImputes the data in a DataFrame or DataTable by imputing the values 1 column at a time; if this is not the desired behaviour custom imputor methods should overload this method.\n\nArguments\n\nimp::Imputor: the Imputor method to use\nctx::Context: the contextual information for missing data\ndata::Table: the data to impute\n\nReturns\n\nTable: the input data with values imputed\n\n\n\n"
},

{
    "location": "api/imputors.html#Impute.impute!-Union{Tuple{Impute.Imputor,Impute.Context,AbstractArray{T,2}}, Tuple{T}} where T",
    "page": "Imputors",
    "title": "Impute.impute!",
    "category": "Method",
    "text": "impute!{T<:Any}(imp::Imputor, ctx::Context, data::AbstractArray{T, 2})\n\nImputes the data in a matrix by imputing the values 1 column at a time; if this is not the desired behaviour custom imputor methods should overload this method.\n\nArguments\n\nimp::Imputor: the Imputor method to use\nctx::Context: the contextual information for missing data\ndata::AbstractArray{T, 2}: the data to impute\n\nReturns\n\nAbstractArray{T, 2}: the input data with values imputed\n\n\n\n"
},

{
    "location": "api/imputors.html#Impute.Imputor",
    "page": "Imputors",
    "title": "Impute.Imputor",
    "category": "Type",
    "text": "Imputor\n\nAn imputor stores information about imputing values in AbstractArrays, DataFrames and DataTables. New imputation methods are expected to sutype Imputor and, at minimum, implement the impute!{T<:Any}(imp::<MyImputor>, ctx::Context, data::AbstractArray{T, 1}) method.\n\n\n\n"
},

{
    "location": "api/imputors.html#",
    "page": "Imputors",
    "title": "Imputors",
    "category": "page",
    "text": "Modules = [Impute]\nPages = [\"imputors.jl\"]\nOrder = [:module, :constant, :type, :function]"
},

{
    "location": "api/imputors.html#Impute.impute!-Tuple{Impute.Drop,Impute.Context,Union{DataFrames.DataFrame, DataTables.DataTable}}",
    "page": "Imputors",
    "title": "Impute.impute!",
    "category": "Method",
    "text": "impute!(imp::Drop, ctx::Context, data::Table)\n\nFinds the missing rows in the DataFrame or DataTable and deletes them.\n\nNOTE: this isn't quite as fast as dropnull in DataTables as we're using an arbitrary missing function rather than using the precomputed dt.isnull vector of bools.\n\nArguments\n\nimp::Drop: this Imputor method\nctx::Context: contextual information for missing data\ndata::Table: the data to impute\n\nReturns\n\nTable: our data with the missing rows removed.\n\n\n\n"
},

{
    "location": "api/imputors.html#Impute.impute!-Union{Tuple{Impute.Drop,Impute.Context,AbstractArray{T,1}}, Tuple{T}} where T",
    "page": "Imputors",
    "title": "Impute.impute!",
    "category": "Method",
    "text": "impute!{T<:Any}(imp::Drop, ctx::Context, data::AbstractArray{T, 1})\n\nUses filter! to remove missing elements from the array.\n\nArguments\n\nimp::Drop: this Imputor method\nctx::Context: contextual information for missing data\ndata::AbstractArray{T, 1}: the data to impute\n\nReturns\n\nAbstractArray{T, 1}: our data array with missing elements removed\n\n\n\n"
},

{
    "location": "api/imputors.html#Impute.impute!-Union{Tuple{Impute.Drop,Impute.Context,AbstractArray{T,2}}, Tuple{T}} where T",
    "page": "Imputors",
    "title": "Impute.impute!",
    "category": "Method",
    "text": "impute!{T<:Any}(imp::Drop, ctx::Context, data::AbstractArray{T, 2})\n\nFinds the missing rows in the matrix and uses a mask (Array{Bool, 1}) to return the data with those rows removed. Unfortunately, the mask approach requires copying the matrix.\n\nNOTES (or premature optimizations):\n\nWe use view, but this will change the type of the data by returning a SubArray\nWe might be able to do something clever by:\nreshaping the data to a vector\nrunning deleteat! for the appropriate indices and\nreshaping the data back to the desired shape.\n\nArguments\n\nimp::Drop: this Imputor method\nctx::Context: contextual information for missing data\ndata::AbstractArray{T, 2}: the data to impute\n\nReturns\n\nAbstractArray{T, 2}: a new matrix with missing rows removed\n\n\n\n"
},

{
    "location": "api/imputors.html#Impute.Drop",
    "page": "Imputors",
    "title": "Impute.Drop",
    "category": "Type",
    "text": "Drop <: Imputor\n\nRemoves missing values from the AbstractArray, DataFrame or DataTable provided.\n\n\n\n"
},

{
    "location": "api/imputors.html#Drop-1",
    "page": "Imputors",
    "title": "Drop",
    "category": "section",
    "text": "Modules = [Impute]\nPages = [\"drop.jl\"]\nOrder = [:module, :constant, :type, :function]"
},

{
    "location": "api/imputors.html#Impute.impute!-Union{Tuple{Impute.Fill,Impute.Context,AbstractArray{T,1}}, Tuple{T}} where T",
    "page": "Imputors",
    "title": "Impute.impute!",
    "category": "Method",
    "text": "impute!{T<:Any}(imp::Fill, ctx::Context, data::AbstractArray{T, 1})\n\nComputes the fill value if imp.value is a Function (i.e., imp.value(drop(copy(data)))) and replaces all missing values in the data with that value.\n\n\n\n"
},

{
    "location": "api/imputors.html#Impute.Fill",
    "page": "Imputors",
    "title": "Impute.Fill",
    "category": "Type",
    "text": "Fill <: Imputor\n\nFills in the missing data with a specific value.\n\nFields\n\nvalue::Any: A scalar missing value or a function that returns the a scalar if   passed the data with missing data removed (e.g, mean)\n\n\n\n"
},

{
    "location": "api/imputors.html#Impute.Fill-Tuple{}",
    "page": "Imputors",
    "title": "Impute.Fill",
    "category": "Method",
    "text": "Fill() -> Fill\n\nBy default Fill() will use the mean of the existing values as the fill value.\n\n\n\n"
},

{
    "location": "api/imputors.html#Fill-1",
    "page": "Imputors",
    "title": "Fill",
    "category": "section",
    "text": "Modules = [Impute]\nPages = [\"fill.jl\"]\nOrder = [:module, :constant, :type, :function]"
},

{
    "location": "api/imputors.html#Impute.impute!-Union{Tuple{Impute.Interpolate,Impute.Context,AbstractArray{T,1}}, Tuple{T}} where T",
    "page": "Imputors",
    "title": "Impute.impute!",
    "category": "Method",
    "text": "impute!{T<:Any}(imp::Interpolate, ctx::Context, data::AbstractArray{T, 1})\n\nUses linear interpolation between existing elements of a vector to fill in missing data.\n\nWARNING: Missing values at the head or tail of the array cannot be interpolated if there are no existing values on both sides. As a result, this method does not guarantee that all missing values will be imputed.\n\n\n\n"
},

{
    "location": "api/imputors.html#Impute.Interpolate",
    "page": "Imputors",
    "title": "Impute.Interpolate",
    "category": "Type",
    "text": "Interpolate <: Imputor\n\nPerforms linear interpolation between the nearest values in an vector.\n\n\n\n"
},

{
    "location": "api/imputors.html#Interpolate-1",
    "page": "Imputors",
    "title": "Interpolate",
    "category": "section",
    "text": "Modules = [Impute]\nPages = [\"interp.jl\"]\nOrder = [:module, :constant, :type, :function]"
},

{
    "location": "api/imputors.html#Impute.impute!-Union{Tuple{Impute.LOCF,Impute.Context,AbstractArray{T,1}}, Tuple{T}} where T",
    "page": "Imputors",
    "title": "Impute.impute!",
    "category": "Method",
    "text": "impute!{T<:Any}(imp::LOCF, ctx::Context, data::AbstractArray{T, 1})\n\nIterates forwards through the data and fills missing data with the last existing observation.\n\nWARNING: missing elements at the head of the array may not be imputed if there is no existing observation to carry forward. As a result, this method does not guarantee that all missing values will be imputed.\n\nUsage\n\n\n\n\n\n"
},

{
    "location": "api/imputors.html#Last-Observation-Carried-Forward-(LOCF)-1",
    "page": "Imputors",
    "title": "Last Observation Carried Forward (LOCF)",
    "category": "section",
    "text": "Modules = [Impute]\nPages = [\"locf.jl\"]\nOrder = [:module, :constant, :type, :function]"
},

{
    "location": "api/imputors.html#Impute.impute!-Union{Tuple{Impute.NOCB,Impute.Context,AbstractArray{T,1}}, Tuple{T}} where T",
    "page": "Imputors",
    "title": "Impute.impute!",
    "category": "Method",
    "text": "impute!{T<:Any}(imp::NOCB, ctx::Context, data::AbstractArray{T, 1})\n\nIterates backwards through the data and fills missing data with the next  existing observation.\n\nWARNING: missing elements at the tail of the array may not be imputed if there is no existing observation to carry backward. As a result, this method does not guarantee that all missing values will be imputed.\n\nUsage\n\n\n\n\n\n"
},

{
    "location": "api/imputors.html#Impute.NOCB",
    "page": "Imputors",
    "title": "Impute.NOCB",
    "category": "Type",
    "text": "NOCB <: Imputor\n\nFills in missing data using the Next Observation Carried Backward (NOCB) approach.\n\n\n\n"
},

{
    "location": "api/imputors.html#Next-Observation-Carried-Backward-(NOCB)-1",
    "page": "Imputors",
    "title": "Next Observation Carried Backward (NOCB)",
    "category": "section",
    "text": "Modules = [Impute]\nPages = [\"nocb.jl\"]\nOrder = [:module, :constant, :type, :function]"
},

{
    "location": "api/imputors.html#Impute.impute!-Tuple{Impute.Chain,Function,Union{AbstractArray, DataFrames.DataFrame, DataTables.DataTable}}",
    "page": "Imputors",
    "title": "Impute.impute!",
    "category": "Method",
    "text": "impute!(imp::Chain, missing::Function, data::Dataset; limit::Float64=0.1)\n\nCreates a Context and runs the Imputors on the supplied data.\n\nArguments\n\nimp::Chain: the chain to run\nmissing::Function: the missing function to use in the Context to pass to the Imputors\ndata::Dataset: our data to impute\nlimit::Float64: the missing data ration limit/threshold\n\nReturns\n\nDataset: our imputed data\n\n\n\n"
},

{
    "location": "api/imputors.html#Impute.impute!-Tuple{Impute.Chain,Union{AbstractArray, DataFrames.DataFrame, DataTables.DataTable}}",
    "page": "Imputors",
    "title": "Impute.impute!",
    "category": "Method",
    "text": "impute!(imp::Chain, data::Dataset; limit::Float64=0.1)\n\nInfers the missing data function from the data and passes that to impute!(imp::Chain, missing::Function, data::Dataset; limit::Float64=0.1).\n\nArguments\n\nimp::Chain: the chain to run\ndata::Dataset: our data to impute\nlimit::Float64: the missing data ration limit/threshold\n\nReturns\n\nDataset: our imputed data\n\n\n\n"
},

{
    "location": "api/imputors.html#Impute.Chain",
    "page": "Imputors",
    "title": "Impute.Chain",
    "category": "Type",
    "text": "Chain <: Imputor\n\nRuns multiple Imputors on the same data in the order they're provided.\n\nFields\n\nimputors::Array{Imputor}\n\n\n\n"
},

{
    "location": "api/imputors.html#Impute.Chain-Tuple{Vararg{Impute.Imputor,N} where N}",
    "page": "Imputors",
    "title": "Impute.Chain",
    "category": "Method",
    "text": "Chain(imputors::Imputor...) -> Chain\n\nCreates a Chain using the Imputors provided (ordering matters).\n\n\n\n"
},

{
    "location": "api/imputors.html#Chain-1",
    "page": "Imputors",
    "title": "Chain",
    "category": "section",
    "text": "Modules = [Impute]\nPages = [\"chain.jl\"]\nOrder = [:module, :constant, :type, :function]"
},

{
    "location": "api/utils.html#Base.convert-Union{Tuple{Type{T},Nullable}, Tuple{T}} where T",
    "page": "Utilities",
    "title": "Base.convert",
    "category": "Method",
    "text": "convert{T<:Any}(::Type{T}, x::Nullable)\n\nConverts the value of a Nullable to the specified type. Needed for casting Nullables to Reals for use in UnitRanges\n\n\n\n"
},

{
    "location": "api/utils.html#Base.filter!-Tuple{Function,Union{DataArrays.DataArray, NullableArrays.NullableArray}}",
    "page": "Utilities",
    "title": "Base.filter!",
    "category": "Method",
    "text": "filter!(f::Function, a::Union{NullableArray, DataArray})\n\nAllows filtering on NullableArrays and DataArrays, this is pretty much copy-paste from base julia, but they only supports filter! on Array{T, 1} since not all AbstractVectors will implement @deleteat!.\n\n\n\n"
},

{
    "location": "api/utils.html#RDatasets.dataset-Tuple{Module,AbstractString,AbstractString}",
    "page": "Utilities",
    "title": "RDatasets.dataset",
    "category": "Method",
    "text": "dataset(m::Module, package_name::AbstractString, Dataset_name::AbstractString)\n\nFunction for optional loading of test RDatasets into either DataFrames or DataTables\n\n\n\n"
},

{
    "location": "api/utils.html#",
    "page": "Utilities",
    "title": "Utilities",
    "category": "page",
    "text": "Modules = [Impute]\nPages = [\"utils.jl\"]"
},

]}
