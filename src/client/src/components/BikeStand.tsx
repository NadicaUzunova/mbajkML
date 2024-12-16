import React, { FC, useContext, useEffect, useState } from "react";
import './bikestand.css';
import { BikeStandContext } from "../context/bikeStandsContext";
import BikeIcon from '../assets/bike-icon.png';
import { mbajkApi } from "../utils/axios";
import { v4 as uuid } from 'uuid';

type BikeStandProps = {};

type DataRow = (string | number)[]; // Type for each row of old data

const BikeStand: FC<BikeStandProps> = () => {
    const [hours, setHours] = useState<number[]>([]);
    const [data, setData] = useState<number | null>(null);
    const [predictions, setPredictions] = useState<number[]>([]);
    const [isLoading, setIsLoading] = useState<boolean>(true);
    const { currentBikeStand } = useContext(BikeStandContext);

    /**
     * Calculate the next 6 hours for predictions display.
     */
    const calculateNextSixHours = () => {
        const currentDate = new Date();
        const currentHour = currentDate.getHours();
        const nextHours = [];

        for (let i = 1; i <= 7; i++) {
            const nextHour = (currentHour + i) % 24;
            nextHours.push(nextHour);
        }

        setHours(nextHours);
    };

    /**
     * Fetch predictions and current bike stand data.
     */
    const fetchPredictions = async () => {
        if (currentBikeStand) {
            setIsLoading(true);
            try {
                console.log("Fetching data for:", currentBikeStand);

                // Fetch live data from the backend
                const liveDataRes = await mbajkApi.post('http://localhost:8000/live-data', {
                    location: currentBikeStand.name || currentBikeStand.location,
                });

                const liveData = liveDataRes.data;
                console.log("Live data fetched:", liveData);

                const currentBikes = liveData.available_bikes || 0;
                setData(currentBikes);

                // Fetch old data from the backend
                const oldDataRes = await mbajkApi.post('http://localhost:8000/data', {
                    location: currentBikeStand.name || currentBikeStand.location,
                });

                const oldData: DataRow[] = oldDataRes.data.data;
                console.log("Old data fetched:", oldData);

                // Clean and process old data
                const cleanOldData = oldData.map((row: DataRow) => {
                    return row.map(value => {
                        if (typeof value === 'string' && !isNaN(Number(value))) {
                            return Number(value); // Convert numeric strings to numbers
                        }
                        return value === null || value === undefined ? 0 : value; // Replace null/undefined with 0
                    });
                });

                console.log("Cleaned old data:", cleanOldData);

                // Fetch predictions from the backend
                const predictionRes = await mbajkApi.post('http://localhost:8000/mbajk/predict', {
                    location: currentBikeStand.location,
                    data: cleanOldData,
                });

                const predictions = predictionRes.data.predictions;
                console.log("Predictions fetched:", predictions);

                setPredictions(predictions);
            } catch (error) {
                console.error("Error fetching predictions:", error);
            } finally {
                setIsLoading(false);
            }
        }
    };

    /**
     * Fetch predictions on component mount and when the bike stand changes.
     */
    useEffect(() => {
        calculateNextSixHours();
        fetchPredictions();
    }, [currentBikeStand]);

    return (
        <div className="bike-stand-menu-container">
            <div className="bike-stand-inner-container">
                {currentBikeStand?.location && (
                    <>
                        {/* Display the station name */}
                        <div className="bike-stand-menu-title">
                            {currentBikeStand.location} {/* Ensure correct display of š, č, ž */}
                        </div>

                        {/* Display current state */}
                        <div className="bike-stand-current-state-title">Trenutno stanje:</div>
                        <div className="bike-stand-current-state">
                            {isLoading ? (
                                <div className="loading-spinner"></div> // Show a loading spinner while waiting
                            ) : (
                                <div className="bike-stand-current-state-number">
                                    {data !== null ? data : "N/A"}
                                </div>
                            )}
                            <div className="bike-stand-current-state-image">
                                <img src={BikeIcon} width="20px" height="20px" alt="Bike icon" />
                            </div>
                        </div>

                        {/* Display predictions */}
                        <h3>Napovedi</h3>
                        <div className="date-grid-container">
                            {/* Display hours */}
                            {hours.map((hour, index) => (
                                <div key={`hour-${index}`} className="date-grid-item">
                                    {hour}:00
                                </div>
                            ))}

                            {/* Display predictions */}
                            {predictions.map((prediction) => (
                                <div key={uuid()} className="date-grid-item">
                                    <div className="date-grid-item-text">{prediction}</div>
                                    <div className="date-grid-item-image">
                                        <img src={BikeIcon} width="20px" height="20px" alt="Bike icon" />
                                    </div>
                                </div>
                            ))}
                        </div>
                    </>
                )}
            </div>
        </div>
    );
};

export default BikeStand;