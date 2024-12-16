# Stage 1: Build react app
FROM node:16-alpine as build

WORKDIR /app

COPY package*.json ./

RUN npm install

COPY . .

# Load environment variable from Docker Compose
ARG REACT_APP_BACKEND_API_URL
ENV REACT_APP_BACKEND_API_URL=$REACT_APP_BACKEND_API_URL

RUN npm run build

# Stage 2: Serve react app
FROM node:16-alpine

WORKDIR /app

COPY --from=build /app/build ./build

# Define the port for the application
ARG PORT=3000
ENV PORT=$PORT

EXPOSE $PORT

CMD npm install -g serve && serve -s build -l $PORT