"""Load profile for the Module 2 milestone.

Every virtual user logs in as the same demo `admin`, so they share one
per-user rate-limit bucket (5/minute on /run and /ask). Expect the failure
column to fill with 429s under any real load - that is the limiter working.
/run and /ask also call the real graph, so their latency includes the model;
read /health P95/P99 for the app's own overhead as concurrency climbs.
"""

import os

from locust import HttpUser, between, task

# Demo credentials for the local user store. Override to match the running
# stack's DEMO_USERNAME / DEMO_PASSWORD; the fallback matches .env.mac.example.
USERNAME = os.getenv("DEMO_USERNAME", "admin")
PASSWORD = os.getenv("DEMO_PASSWORD", "locust-demo")


class EnterpriseUser(HttpUser):
    wait_time = between(1, 2)

    def on_start(self):
        r = self.client.post(
            "/api/v1/token",
            data={"username": USERNAME, "password": PASSWORD},
        )
        r.raise_for_status()
        self.token = r.json()["access_token"]

    @property
    def auth(self):
        return {"Authorization": f"Bearer {self.token}"}

    @task(3)
    def ask(self):
        # Same q every time -> served from Redis after the first hit.
        self.client.get("/api/v1/ask", params={"q": "status report"}, headers=self.auth)

    @task(1)
    def run_agent(self):
        # Not `run` - HttpUser.run() is Locust's internal loop; overriding it errors.
        self.client.post(
            "/api/v1/run",
            json={
                # request_id is a UUID auto-generated when omitted.
                "agent_id": "load_test",  # must match ^[a-zA-Z0-9_-]+$
                "task_description": "Summarise the most recent deployment.",  # 10-500 chars
            },
            headers=self.auth,
        )

    @task(2)
    def health(self):
        # No auth, no rate limit - the load-balancer's view of the service.
        self.client.get("/health")
