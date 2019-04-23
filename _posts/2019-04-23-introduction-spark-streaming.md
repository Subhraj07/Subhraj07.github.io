---
layout: post
title:  "Introduction to Spark Streaming"
date:   2019-04-23 11:00:00
tags: spark Streaming
language: EN
---
Spark permet de traiter des données qui sont figées à un instant _T_. Grâce au module Spark Streaming, il est possible de traiter des flux de données qui arrivent en continu, et donc de traiter ces données au fur et à mesure de leur arrivée.

# Model of micro-batches

With Spark Streaming, a context is initialized with a duration. The framework will accumulate data during this time and then produce a small RDD ( Resilient Distributed Dataset , see Introduction to Apache Spark ). This accumulation / generation cycle of RDD will recur until the program is stopped. We are talking here about micro-batches as opposed to a treatment of events one by one.

<p style="text-align:center;"><img src="/images/spark-streaming-micro-batches.png" style="width:75%"></p>

Spark Streaming is opposed here to Apache Storm (https://storm.apache.org/) : Storm offers a real-time treatment of events while Spark Streaming will add a delay between the arrival of a message and its processing.

This difference in treatment, however, allows Spark Streaming to offer a guarantee of message processing in exactly once in normal conditions (each message is delivered once and only to the program, without loss of messages), and at least once in degraded conditions. (A message can be delivered several times, but still without losses). Storm allows you to set the guarantee level, but to optimize performance, the mode at most once (each message is delivered at most once but losses are possible) must be used.

Finally, and this is the main advantage of **Spark Streaming over Storm, the **Spark Streaming API is identical to the classic Spark API . It is thus possible to manipulate data streams in the same way as manipulated frozen data.


# Data sources

Spark Streaming is intended to process data that arrive continuously, it is necessary to choose a suitable data source. We will therefore tend to prefer sources opening a network socket and remaining listening. Basic, we can thus use:

- TCP socket (via `sc.socketStream` ou `sc.socketTextStream`)
- messages from [Kafka](http://kafka.apache.org/)
- logs from [Flume](http://flume.apache.org/)
- files from HDFS (to monitor the creation of new files only)
- an MQ queue (type [ZeroMQ](http://zeromq.org/))
- Tweets from Twitter (utilise l'API [Twitter4J](http://twitter4j.org/en/index.html))


It is also possible to implement a custom data source by extending the [Receiver](http://spark.apache.org/docs/latest/api/java/org/apache/spark/streaming/receiver/Receiver.html) class.