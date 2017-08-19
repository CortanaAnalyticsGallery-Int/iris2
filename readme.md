# Classifying Iris

Run iris_sklearn.py in local environment.
```
$ az ml execute start -c local iris_sklearn.py
```

Run iris_sklearn.py in a local Docker container.
```
$ az ml execute start -c docker iris_sklearn.py
```

Create myvm.compute file to point to a remote VM
```
$ az ml computecontext attach --name <myvm> --address <ip address or FQDN> --username <username> --password <pwd>
```

Run iris_pyspark.py in a Docker container (with Spark) in a remote VM:
```
$ az ml execute start -c myvm iris_pyspark.py
```

Create myhdi.compute to point to an HDI cluster
```
$ az ml computecontext attach --name <myhdi> --address <ip address or FQDN of the head node> --username <username> --password <pwd> --cluster
```

Run it in a remote HDInsight cluster:
```
$ az ml execute start -c myhdi iris_pyspark.py
```
For more details on configuring execution targets, go to: http://aka.ms/vienna-docs-exec
