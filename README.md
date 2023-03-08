# AIRBNB PRICE PREDICTION ASSIGNMENT

This repository is intended for the group members of the *Advanced Analytics in a Big Data World* course to collaborate for the first assignment (Airbnb price prediction).

- I have included a *create_venv.ps1* file that creates a virtual environment and installs the packages included in the *requirements.txt* file automatically.

- There are several packages in the requirements file, since the plan is to do some experiments with packages like *pycaret* (has a lot of dependencies, hence the big requirements file) and *torch*. The basic ones like *pandas*, *numpy* and *sklearn* are already included.

---

The data set has the following features:

- property_id: unique identifier for the property
- property_name: textual name (title) of the property
- property_summary, property_space, property_desc, property_neighborhood, property_notes,propertytransit,propertyaccess,propertyinteraction,propertyrules`: "free text" fields where the host provides a summary, description of the space, general description, description of the neighborhood, additional notes, notes about transit (accessibility), how to interact with the host and house rules. You don't need to use these, but you can
- property_zipcode, property_lat, property_lon: zip code, latitude and longitude of the property
- property_type, property_room_type: type of the property and room
- property_max_guests: maximum number of guests that can stay per night
- property_bathrooms, property_bedrooms, property_beds, property_bed_type: number of bathrooms, bedrooms, beds and type of bed
- property_amenities: other amenities provided. Note that this is a comma separated list (there are others like this as well), so you will need to preprocess this into appropriate features
- property_sqfeet: square feet of the property
- property_scraped_at: when the information of the property was scraped
- property_last_updated: when (relative the point at which property was scraped) the property info was last updated
- host_id: unique identifier of the host
- host_since: how long the offering person has been a host
- host_location, host_about, host_response_time, host_response_rate: location of the host, description of the host, how long it takes for them to response and how often they respond the questions
- host_nr_listings, host_nr_listings_total: number of listings the host has (and had in total)
- host_verified: which verification schemes the host has enabled
- booking_price_covers: how many people does the price per night cover (can be lower than the max. guests in which case an additional fee is asked per extra person, which is not provided here)
- booking_min_nights, booking_max_nights: min. and max. number of nights you can book in a single booking
- booking_availability_30, booking_availability_60, booking_availability_90, booking_availability_365: availability for the next month, two months, three months and year (based on point of reference when the property was scraped)
- booking_cancel_policy: cancellation policy used
- reviews_num, reviews_first, reviews_last, reviews_rating: number of reviews, data of first and last review, and average rating
- reviews_acc, reviews_cleanliness, reviews_checkin, reviews_communication, reviews_location, reviews_value: separate ratings for accuracy, cleanliness, and so on
- reviews_per_month: number of reviews per month on average
- extra: additional list of comma separated information on host or property
- target: price per night (only in train set)