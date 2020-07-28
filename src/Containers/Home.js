import React, { useState, useEffect } from "react";
import { gql } from "apollo-boost";
import { useQuery } from '@apollo/react-hooks';

import { config } from "../config";
import Loader from '../Components/Loader'
import EventCard from '../Components/EventCard'
import { Button, Container, Col, Row } from 'reactstrap'
import styled from "styled-components";
import heroImg from '../img/hero-img.png';
import aboutImg from '../img/about.png';
import { Link } from "react-router-dom";

const GET_POSTS = gql`
{
  repository(owner: "${config.githubUserName}", name: "${config.githubRepo}") {
    issues(first: 3, states: OPEN, filterBy: { labels: "event" }, orderBy: {direction: DESC, field: CREATED_AT}) {
      nodes {
        title
        body
        number
      }
    }
  }
}
`

const Home = () => {
  const [posts, setPosts] = useState([]);
  const { loading, error, data } = useQuery(GET_POSTS);

  useEffect(() => {
    if (!loading) {
      if (error) {
        console.error(error)
      }
      if (data) {
        setPosts(data?.repository?.issues?.nodes)
      }
    }
  }, [loading, error, data]);

  return (
    <>
      <LandingHome>
        <Container fluid className="h-100 d-flex align-items-center">
          <Row>
            <Col sm="12" lg={{ offset: 2, size: 8 }}>
              <h1 className="text-primary display-3 font-weight-bold">For The Ones In Need</h1>
              <p className="mt-3 mb-4 text-secondary">A Bengal based NGO, with a vision to make a difference in society for the civic benefit.</p>
              <Link to="/contact">
                <Button className="rounded-0 px-4" color="success" size="lg">
                  Contact
                </Button>
              </Link>
            </Col>
          </Row>
        </Container>
      </LandingHome>
      <div style={{ background: "#f8fcff", padding: "100px 0" }}>
        <Container fluid>
          <Row>
            <Col className="d-flex justify-content-center flex-column" sm="12" lg={{ offset: 1, size: 5 }}>
              <h5 className="text-success font-weight-bold mb-3">About Our Foundation</h5>
              <h1 className="text-primary font-weight-bold mb-4">We Are In A Mission To  Help The Helpless</h1>
              <p className="text-secondary">Lorem ipsum dolor sit amet, consectetur adipisicing elit,mod tempor incididunt ut labore et dolore magna aliqua. Utnixm, quis nostrud exercitation ullamc.</p>
              <p className="text-secondary">Lorem ipvsum dolor sit amext, consectetur adipisicing elit, smod tempor incididunt ut labore et dolore.</p>
              <Link to="/about">
                <Button className="rounded-0 px-4" color="success" size="lg">
                  About Us
                </Button>
              </Link>
            </Col>
            <Col className="text-center mt-4 mt-lg-0" sm="12" lg="5">
              <AboutImg src={aboutImg} alt="" />
            </Col>
          </Row>
        </Container>
      </div>
      <div style={{ padding: "100px 0" }}>
        <Container fluid>
          <Row>
            <Col className="text-center" sm="12" md={{ offset: 2, size: 8 }} lg={{ offset: 3, size: 6 }}>
              <h5 className="text-success font-weight-bold mb-3">Our Recent Efforts</h5>
              <h1 className="text-primary font-weight-bold mb-4">Explore our latest causes that we worked on</h1>
            </Col>
          </Row>
          <Row className="justify-content-center">
            {
              loading
                ? <Loader />
                : posts.map((v, i) => {
                  return <EventCard data={v} key={i} />;
                })
            }
          </Row>
          <div className="text-center mt-3">
            <Link to="/events">
              <Button className="rounded-0 px-4" color="success" size="sm">
                View More +
              </Button>
            </Link>
          </div>
        </Container>
      </div>
    </>
  );
}

const LandingHome = styled.div`
  height: 100vh;
  background-image: url(${heroImg});
  background-repeat: no-repeat;
  background-position: center center;
  background-size: cover;
`

const AboutImg = styled.img`
  @media (max-width: 991.98px) { 
    width: 100%;
  }
`

export default Home;
