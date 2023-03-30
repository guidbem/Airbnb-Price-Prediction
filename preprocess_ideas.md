### Most Likely Non-usable features:

- property_id;
- host_id;
- property_name;
- host_location;
- host_response_time; (directly correlated with the response rate)
- host_since;
- host_nr_listings_total (it is the same feature as host_nr_listings);
- property_zipcode (we already have latitude and longitude, easier to do a smaller clustering on those than on the 41 different zipcode values);
- property_last_updated & property_scraped_at (outdated);
- reviews_first & reviews_last (not relevant);
- property_sqfeet (too many missing values, might have to be dropped);
- property_neighborhood;
- property_notes; 
- property_transit; 
- property_access;
- property_interaction;
- property_rules;


- property_desc; (ONLY 3 NAs)**
    - Maybe for the *property_desc*, we can use the size of the text as a feature (larger texts are usually viewed as more positive and can lead to people trusting it more and being ok to pay a higher price). (USE LATER MAYBE)

These features probably don't influence at all the price of the property on Airbnb.

### Hard to Use Text Features:

- property_summary;
- property_space; 
- host_about;

These Features contain a lot of text, not following a pattern and user-typed. Hard to automate feature extraction from these. 

Suggestion: We can use the ones that have a minimum number** of missing values and create a feature for each called *has_(feature name)* that would be a binary feature stating if the information is missing or not. DO THIS

### Easy to Use Text/Categorical/Date Features:

Simply encode:

- property_type;
- property_room_type;
- property_bed_type;
- booking_cancel_policy;

Process:

- property_amenities (perhaps not so easy, list of several amenities available in the property);
COUNT THE NUMBER OF AMENITIES IN EACH OBSERVATION
IMPUTE MISSING VALUES

- host_verified 
(count the number of different methods and apply a log transform, DECIDED TO TRY WITHOUT THE LOG TRANSFORM AND ONLY STANDARDIZE THE DATA);

- extra (maybe not so easy, but it seems to follow a pattern of positive traits separated by commas);
    Which to use:
        - Host is Superhost;
        - Is Location Exact;
        - Instant Bookable;


Some of these features can probably be directly used as categorical features, while others can be transformed in other features.

### Easy to Use Numeric Features:

- property_lat & property_lon (use k means clustering to separate the locations per zone as categories);

- property_max_guests;
- property_bathrooms;
- property_bedrooms;
- property_beds;
- host_response_rate;
- host_nr_listings;
- booking_price_covers;
- booking_min_nights;
- booking_max_nights;
- booking_availability_30;
- booking_availability_60;
- booking_availability_90;
- booking_availability_365;
- reviews_num;
- reviews_per_month;

SCALE AND IMPUTATION

APPLY PCA TO THESE
- reviews_rating;
- reviews_acc;
- reviews_cleanliness;
- reviews_checkin;
- reviews_communication;
- reviews_location;
- reviews_value;


These are numerical features that are pretty much ready to be used, just need to be treated for missing values.

---

- remove observations that have an abs z score higher than 2.58 (99% of the data) on target

- Bin the property_beds into:
    - 1 to 6 beds categories
    - 7+ beds
    - Use target encoding on it

- property max_guests should be between 1x and 2x the number of beds, truncate to upper or lower otherwise

- Bin the property_max_guests into:
    - 1 to 6 max guests categories
    - 7+ guests
    - Use target encoding on it
