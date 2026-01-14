# CI/CD Pipeline Documentation

## Overview

This project uses GitHub Actions for continuous deployment to AWS EC2. Every push to the `main` branch automatically triggers a deployment.

## Architecture

```
Developer pushes code → GitHub Actions → SSH to EC2 → Pull code → Build Docker → Deploy
```

## Workflow Steps

1. **Trigger**: Push to `main` branch or manual workflow dispatch
2. **Checkout**: Clone the latest code from GitHub
3. **SSH Setup**: Configure SSH keys for EC2 access
4. **Deploy**: Connect to EC2 and execute deployment script
5. **Build**: Create new Docker image from latest code
6. **Deploy**: Stop old container, start new container
7. **Verify**: Check that the container is running successfully

## Required GitHub Secrets

The following secrets must be configured in GitHub repository settings:

| Secret Name | Description | Example |
|-------------|-------------|---------|
| `SSH_PRIVATE_KEY` | Private SSH key for EC2 access | `-----BEGIN OPENSSH PRIVATE KEY-----...` |
| `EC2_HOST` | Elastic IP of your EC2 instance | `52.45.123.78` |
| `EC2_USER` | SSH username for EC2 | `ec2-user` or `ubuntu` |
| `OPENAI_API_KEY` | OpenAI API key for the application | `sk-proj-...` |

## Setup Instructions

### 1. Generate Deployment SSH Key (On EC2)

```bash
# SSH into your EC2 instance
ssh -i ~/.ssh/bristolbot-key.pem ec2-user@YOUR_IP

# Generate new SSH key for deployment
ssh-keygen -t ed25519 -C "github-actions-deploy" -f ~/.ssh/github_actions_deploy

# Add public key to authorized_keys
cat ~/.ssh/github_actions_deploy.pub >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys

# Display private key (copy this entire output)
cat ~/.ssh/github_actions_deploy
```

### 2. Configure GitHub Secrets

1. Go to your GitHub repository
2. Click **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. Add all 4 secrets listed above

### 3. Create Workflow File

The workflow file is located at `.github/workflows/deploy.yml`. It's already configured and ready to use.

### 4. Test the Pipeline

```bash
# Make a small change (e.g., update README)
echo "# Testing CI/CD" >> README.md

# Commit and push
git add README.md
git commit -m "test: trigger CI/CD pipeline"
git push origin main

# Go to GitHub → Actions tab to watch the deployment
```

## Manual Deployment

You can also trigger deployments manually:

1. Go to GitHub repository → **Actions** tab
2. Select "Deploy to AWS EC2" workflow
3. Click **Run workflow** → **Run workflow**

## Monitoring Deployments

### GitHub Actions Dashboard

- View deployment status: `https://github.com/YOUR_USERNAME/bristolbot/actions`
- Green checkmark ✅ = Success
- Red X ❌ = Failed (click to see logs)

### EC2 Logs

```bash
# SSH into EC2
ssh -i ~/.ssh/bristolbot-key.pem ec2-user@YOUR_IP

# Check container status
docker ps

# View application logs
docker logs bristolbot -f

# Check if deployment script ran
tail -f /var/log/cloud-init-output.log
```

## Troubleshooting

### Pipeline fails at "Create SSH key file"
- **Cause**: SSH_PRIVATE_KEY secret is incorrect
- **Fix**: Re-generate SSH key on EC2 and update GitHub secret

### Pipeline fails at "Deploy to EC2"
- **Cause**: EC2_HOST or EC2_USER is incorrect
- **Fix**: Verify your Elastic IP and username (ec2-user vs ubuntu)

### Container doesn't start after deployment
- **Cause**: Docker build failed or .env missing
- **Fix**: SSH into EC2 and check `docker logs bristolbot`

### App not accessible after deployment
- **Cause**: Container needs time to start
- **Fix**: Wait 30 seconds, then refresh. Check security group has port 8501 open.

## Rollback Procedure

If a deployment breaks the app:

```bash
# SSH into EC2
ssh -i ~/.ssh/bristolbot-key.pem ec2-user@YOUR_IP

# Find previous working image
docker images

# Stop current container
docker stop bristolbot
docker rm bristolbot

# Run previous image
docker run -d -p 8501:8501 --name bristolbot --restart unless-stopped --env-file .env bristolbot:TAG

# OR revert code and re-deploy
cd ~/UniBristol-RAG-Assistant
git log  # Find working commit
git reset --hard COMMIT_HASH
git push -f origin main  # This will trigger new deployment
```

## Best Practices

1. **Test locally first**: Always test changes locally before pushing
2. **Small commits**: Make small, incremental changes
3. **Monitor deployments**: Watch the Actions tab after pushing
4. **Keep secrets secure**: Never commit secrets to code
5. **Use branches**: Create feature branches, merge to main when ready

## Deployment Timeline

Typical deployment takes **3-5 minutes**:
- Checkout code: 10s
- SSH setup: 5s
- Git pull: 10s
- Docker build: 2-3 mins
- Container restart: 30s
- Verification: 10s

## Cost Implications

- **GitHub Actions**: 2,000 free minutes/month (this uses ~5 mins/deployment)
- **EC2**: Runs 24/7, cost unchanged by CI/CD
- **Data Transfer**: Minimal (only code changes, not re-downloading dependencies)

## Future Enhancements

- [ ] Add automated testing before deployment
- [ ] Implement blue-green deployment for zero downtime
- [ ] Add Slack/email notifications on deployment
- [ ] Set up staging environment
- [ ] Add automatic rollback on failure

## Security Notes

- SSH private key is encrypted in GitHub Secrets
- Never share or commit the private key
- Rotate SSH keys periodically (every 90 days)
- Use separate keys for different environments
- Limit SSH access with security groups

## Questions?

If you encounter issues:
1. Check GitHub Actions logs
2. SSH into EC2 and check Docker logs
3. Verify all secrets are correctly configured
4. Ensure EC2 security group allows port 8501

---

**Last Updated**: January 2026
**Maintainer**: Your Name
