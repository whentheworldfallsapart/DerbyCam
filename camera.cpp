#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>
#include <time.h>
using namespace cv;
using namespace std;

int main(int, char **)
{   //Config Parameters
    const unsigned BUF_SIZE = 90; //Buffer Size (number of frames)
    const unsigned REFILL_NUM = 90; //number of frames to refill buffer before next trigger
    const int FRAME_DELAY = 120; //time in ms to delay between frames for slow motion reply
    const double DELAY_TIME_SECS = 1.5; //Time to delay before starting replay
    const int EROSION_SIZE = 6; //the erosion kernel size
    const int MOTION_HIS = 7; //num frame histories to use for motion 
    const int MOTION_DTH = 36; //threshold for motion detection
    const int NUM_PIXELS_DTH = 100; //number of pixels that must pass background and errode to declare motion

    //Simple Buffer
    std::vector<cv::Mat> buffer;
    // Start and end times
    time_t start, end, delay_start, delay_end;
    double delay_secs = 0.0;
    bool replay = false; //flag for starting the replay (depend on motion trigger and delay comlpete)
    bool delay = false; //in the delay state (motion has been triggered but we dont want to replay instantly)
    bool haveMotionBox = false; //flag to get motion box
    unsigned index = 0; //index into the buffer for reading
    unsigned count = 0; //number of frames captured
    unsigned sinceLast = 0; //number of frames since last replay
    unsigned numFrames = 0;
    unsigned outLoop = 0; //size of the backend of the buffer incase we didnt fill it up
    int numPixels = 0; //number of pixels that passed motion after errode
    cv::Rect2d roi; //rect box for crop image
    cv::Mat frame; //the captured and display frame
    cv::Mat cropImg; //the motion frame
    cv::Mat fgMaskMOG2; //the motion frame
    cv::Mat erd; //the eroded motion frame
    cv::Ptr<BackgroundSubtractor> pMOG2; 
    pMOG2 = cv::createBackgroundSubtractorMOG2(MOTION_HIS, MOTION_DTH, false); //MOG2 approach
    cv::Mat element = getStructuringElement(cv::MORPH_CROSS,cv::Size(2 * EROSION_SIZE + 1, 2 * EROSION_SIZE + 1), cv::Point(EROSION_SIZE, EROSION_SIZE));
    //--- INITIALIZE VIDEOCAPTURE
    cv::VideoCapture cap;
    int deviceID = 0;                                         // 0 = open default camera
    int apiID = cv::CAP_V4L;                                  //cv::CAP_ANY;      // 0 = autodetect default API
    // open selected camera using selected API
    cap.open(deviceID + apiID);
    // check if we succeeded
    if (!cap.isOpened()) {
        cerr << "ERROR! Unable to open camera\n";
        return -1;
    }
    //set full screen
    cv::namedWindow("Track Camera", cv::WindowFlags::WINDOW_NORMAL);
    cv::setWindowProperty("Track Camera", cv::WND_PROP_FULLSCREEN, cv::WindowFlags::WINDOW_FULLSCREEN);
    //--- GRAB AND WRITE LOOP
    std::cout << "Start grabbing" << endl << "Press any key to terminate" << endl;
    // Start time
    time(&start);
    //Master Loop
    for (;;) {
        // wait for a new frame from camera and store it into 'frame'
        cap.read(frame);
        // check if we succeeded
        if (frame.empty())  {
            cerr << "ERROR! blank frame grabbed\n";
            break;
        }
        else {

            //if we havnt got the rect box for motion
            if (!(haveMotionBox) & count > 5) {
                cv::putText(frame, "Select Motion ROI", cv::Point(30, 30), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(255, 0, 0), 1, 0);
                roi = cv::selectROI("Track Camera",frame);
                std::cout << "ROI" << roi << std::endl;
                cropImg = frame(roi);
                //apply background subtraction for motion detection
                pMOG2->apply(cropImg, fgMaskMOG2);
                // Apply erosion or dilation on the image to remove false alarms
                cv::erode(fgMaskMOG2, erd, element); 
                numPixels = countNonZero(erd); //check if some pixels or blobs passed the errode
                haveMotionBox = true;
            } else if(haveMotionBox) {
                //std::cout << "ROI" << roi << std::endl;
                cropImg = frame(roi);
                //apply background subtraction for motion detection
                pMOG2->apply(cropImg, fgMaskMOG2);
                // Apply erosion or dilation on the image to remove false alarms
                cv::erode(fgMaskMOG2, erd, element); 
                numPixels = countNonZero(erd); //check if some pixels or blobs passed the errode
            } else {
                numPixels = 0;
            }

            //if the buffer isnt full yet
            if (count < BUF_SIZE) {
                buffer.push_back(frame.clone());
            }
            //else check if the index is at the end of the buffer
            else {
                if (index >= BUF_SIZE) {
                    index = 0; //reset the index to start of buffer to make ring buffer
                }
                buffer[index] = frame.clone();
            }
            count++;
            index++;
            sinceLast++;
        }//end capture/process frame

        //simple state machine
        //if if we had enough pixels pass the errode declare motion
        //check that we had at least 30 frames of videio and that we arent already in the delay state
        if (numPixels > NUM_PIXELS_DTH & sinceLast > REFILL_NUM & !(delay)) {
            //cout << "Pixels Passing : " << numPixels << endl;
            time(&delay_start); //start the delay timer
            delay = true; //transition to the delay state, where we wait before triggering replay
        } else if (delay) {
            //if we are in the delay state get the time and check if the delay is up
            time(&delay_end); 
            numFrames = count; //for fps calc
            delay_secs = difftime(delay_end, delay_start);
            //if the delay is done transition from the delay state to the replay state
            if (delay_secs > DELAY_TIME_SECS) {
                replay = true;
                delay = false;
            }
        }

        //if we are in the replay state, then show the replay once, else show live video
        if (replay) {
            //reset number of frames since replay counter
            sinceLast = 0;
            // End Time
            time(&end); //TODO: this is for FPS, only done once up until replay triggers, need to average fps continuely 
            //cout << "Frame: " << index << endl;
            //get the max size of the replay buffer
            outLoop = count < BUF_SIZE ? count : BUF_SIZE;
            //get the oldests frames from the buffer by starting at the current index 
            for (int k = index; k < outLoop; k++) {
                frame = buffer.at(k);
                cv::putText(frame, "Replay", cv::Point(30, 30), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(0, 0, 255), 1, 0);
                //cout << "FrameFront: " << k << endl;
                cv::imshow("Track Camera", frame);
                if (cv::waitKey(FRAME_DELAY) == 32)
                    break;
            }
            //when the end of the buffer is reach start at the front 
            for (int k = 0; k < index; k++) {
                frame = buffer.at(k);
                cv::putText(frame, "Replay", cv::Point(30, 30), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(0, 0, 255), 1, 0);
                //cout << "FrameBack: " << k << endl;
                cv::imshow("Track Camera", frame);
                if (cv::waitKey(FRAME_DELAY) == 32)
                    break;
            }
            replay = false;
        }
        else
        {   //not in the Replay state so Live frames
            // show live and wait for a key with timeout long enough to show images
            cv::putText(frame, "Live", cv::Point(30, 30), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(255, 0, 0), 1, 0);
            cv::imshow("Track Camera", frame);
            //if(haveMotionBox) {
            //    cv::imshow("FG Mask MOG 2", erd);
            //}
            if (cv::waitKey(1) == 32) {
                break;
            }
        }
    }
    // Time elapsed
    double seconds = difftime(end, start);
    std::cout << "Time taken : " << seconds << " seconds" << endl;
    std::cout << "Frames taken : " << numFrames << endl;
    // Calculate frames per second
    double fps = numFrames / seconds;
    std::cout << "Estimated frames per second : " << fps << endl;
    // the camera will be deinitialized automatically in VideoCapture destructor
    // Release video
    cap.release();
    return 0;
}