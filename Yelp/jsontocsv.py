import sys
import json
import csv
import io
 
   #python ConvertYelp.py yelp_academic_dataset_review.json
   #python ConvertYelp.py yelp_academic_dataset_user.json
   #python ConvertYelp.py yelp_academic_dataset_business.json
   



def filter_and_convert_to_csv(filename):
    if "review" in filename:

        outputFile = open('yelp_academic_dataset_review.csv', 'w')
        fields = ['user_id', 'business_id', 'stars']

        # The "lineterminator='\n' is needed to prevent an extra blank line between each line.
        outputWriter = csv.DictWriter(
            outputFile, fieldnames=fields, lineterminator='\n')
        print("Converting " + filename + " to " +
              "yelp_academic_dataset_review.csv ...")

        # Read line by line, write user_id, business_id, and stars to CSV file
        for line in open(filename, 'r'):
            r = json.loads(line)
            outputWriter.writerow(
                {'user_id': r['user_id'], 'business_id': r['business_id'], 'stars': r['stars']})

    elif "user" in filename:

        outputFile = open('yelp_academic_dataset_user.csv', 'w')
        fields = ['user_id', 'name']

        # The "lineterminator='\n' is needed to prevent an extra blank line between each line.
        outputWriter = csv.DictWriter(
            outputFile, fieldnames=fields, lineterminator='\n')
        print("Converting " + filename + " to " +
              "yelp_academic_user_review.csv ...")

        # Read line by line, write user_id and name to CSV file
        for line in open(filename, 'r'):
            r = json.loads(line)

            # To handle name values with unicode, call "encode" to remove the unicode character.
            # This presents a problem from occurring in "writerow" (which cannot handle unicode well)
            n = r['name']
            n1 = n.encode('ascii', 'ignore')
            outputWriter.writerow({'user_id': r['user_id'], 'name': n1})

    elif "business" in filename:

        outputFile = open('yelp_academic_dataset_business.csv', 'w')
        fields = ['business_id', 'city', 'name',
                  'categories', 'review_count', 'stars']
        outputWriter = csv.DictWriter(
            outputFile, fieldnames=fields, lineterminator='\n')
        print("Converting " + filename + " to " +
              "yelp_academic_dataset_review.csv ...")

        # Read line by line, write relevant fields if the business is a restaurant
        for line in open(filename, 'r'):
            r = json.loads(line)
            categories = str(r['categories'])
            if "Restaurants" in categories:
                # To handle name values with unicode, call "encode" to remove the unicode character.
                # This presents a problem from occurring in "writerow" (which cannot handle unicode well)
                n = r['name']
                n1 = n.encode('ascii', 'ignore')
                c = r['city']
                c1 = c.encode('ascii', 'ignore')

                # Now write the result to a CSV file
                outputWriter.writerow({'business_id': r['business_id'], 'city': c1, 'name': n1, 'categories': r['categories'],
                                       'review_count': r['review_count'], 'stars': r['stars']})

    else:

        print("Error!  Unexpected filename used.")
        exit()

    outputFile.close


def main():
    # This command-line parsing code is provided.
    # Make a list of command line arguments, omitting the [0] element
    # which is the script itself.
    args = sys.argv[1:]

    if not args:
        print('usage: file')
        sys.exit(1)

    filter_and_convert_to_csv(sys.argv[1])


if __name__ == '__main__':
    main()
