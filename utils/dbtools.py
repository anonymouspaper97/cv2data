import mysql.connector

def connect_to_database():
    # Replace with your database connection details
    return mysql.connector.connect(
        host="82.165.141.198",#"217.160.241.60",
        user="crawler",
        password="cRYEQwVxEApPqUPXyQLY",
        database="individual_crawl",
        port = 3306
    )


def personal_data(prof_id):

    conn = conn = connect_to_database()
    cursor = conn.cursor()

    ID = prof_id
    query = f"SELECT  first_name,only_name,title FROM professor WHERE id = {ID}"

    cursor.execute(query)
    rows = cursor.fetchall()

    for row in rows:

        first_name = row[0]

        full_name = row[2]+" "+row[1]

    cursor.close()
    conn.close()

    return first_name,full_name

