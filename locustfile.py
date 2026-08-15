from random import sample

from locust import HttpUser, between, task

QUERIES = ["chocolate", "airfryer", "green beans", "sprouts", "mexican sauce"]
QUERIES_TYPO = ["chocdolate", "airfrier", "gren beans", "sprots", "mexiccan saucs"]


class APIUser(HttpUser):
    wait_time = between(1, 3)

    @task(3)
    def search(self):
        with self.client.get(
            f"/search?query={sample(QUERIES, 1)}", catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Got status code {response.status_code}")

    @task(1)
    def search_typo(self):
        with self.client.get(
            f"/search?query={sample(QUERIES_TYPO, 1)}", catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Got status code {response.status_code}")
