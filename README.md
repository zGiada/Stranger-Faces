# STRANGER FACES: a project about Face Detection and Face Recognition in the Stranger Things World

> PROJECT DEVELOPED FOR DIGITAL FORENSICS COURSE AT @UniPD

------

## Abstract

**Face detection** is a technology that detect and identifies human faces in digital images, and it is used for biometric verification, to reduce crime and prevent violence. Face detection is based on computer vision techniques to advances in machine learning (ML) to increasingly sophisticated neural networks and related technologies: they have the task to find human aspects on a video/image (for example it can start by searching for human eyes, eyebrows, the mouth, nose, etc.).
These algorithms work with ”*feature extraction*” methods: extract salient information that is useful for distinguishing faces of different individuals and is preferably robust with respect to the geometric and photometric variations. To help ensure accuracy, the algorithms need to be trained on large data sets incorporating hundreds of thousands of positive and negative images. The training improves the algorithms’ ability to determine whether there are faces in an image and where they are.

**Face detection** is often confused with face recognition, but a face recognition system is a technology that can identify and compare a person’s identity, determine if the face in two images belongs to the same person, or simply recognize a face from a digital image or face database. It uses biometrics to map face features that are unique to an individual in order to identify and match a person’s identity from a digital image or face database. In fact, face recognition algorithms work by extracting some particular face features and measuring them in such a way as to identify matches with an already known face (typically within a dataset).

## Goals of the project

The goal of this project is to develop a tool capable of detecting faces in an image and video and then being able to recognize their identity (face recognition). 

As a case study, it was decided to apply the face detection and recognition of three actors who are part of the famous TV series called ”Stranger Things”, both in a simple photo and in a video of their interview.

To evaluate the results obtained, general metrics will be observed, dictated by the algorithm that will be developed, by the frameworks used, and by the choice of these images and videos in relation to the dataset that has been created.

## What I used?

- **Python**
- **OpenCV** library
- **Haar-Cascade**
- classification algorithm: **Viola and Jones** Haar Cascade Classifier
- recognition algorithm: **Local Binary Patterns Histograms (LBPH)**