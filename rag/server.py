from fastapi import FastAPI,Query
from client.rq_client import queue
from queues.worker import process_query

app = FastAPI()

@app.get('/')
def health_check():
    return {"status":"Server up and Running!"}


@app.post('/chat')
def chat(query:str = Query(...,description="Chat query of the user")):
    job = queue.enqueue(process_query, query)

    return {"status":"queued", "job_id":job.id}

## For mac os --> export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
## rq worker --with-scheduler

@app.post('/result')
def get_result(job_id:str = Query(..., description="Job Id of the result")):
    job = queue.fetch_job(job_id=job_id)

    if not job:
        return {"result":"Job Id is invalid!"}
    
    result = job.return_value()

    return {"result":result}